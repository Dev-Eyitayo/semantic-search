
"""
Semantic Search Accuracy Evaluator: NDCG@10 & Precision@5

This script evaluates the semantic search system's ranking quality through:
  1. Query embedding using all-MiniLM-L6-v2 (384 dimensions)
  2. Cosine similarity search over Properties using their stored embeddings
  3. Interactive relevance annotation (ground truth: 0-3 scale)
  4. NDCG@10 & Precision@5 metrics using scikit-learn

Usage:
    python eval_ndcg_precision.py

"""

import asyncio
import sys
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime, timezone
from uuid import UUID

# Third-party imports
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.future import select
from sklearn.metrics import ndcg_score, precision_score
from loguru import logger

# Project imports
sys.path.insert(0, '/home/user/Desktop/Projects/fyp/semantic-search')  # Adjust path if needed

from core.config import settings
from db.models.property import Property
from db.models.user import User
from core.enums import PropertyStatus
from services.embedding_service import (
    generate_embedding,
    EMBEDDING_DIMENSIONS,
    get_embedding_model
)


class EvaluationSession:
    """Manages database session and evaluation context."""
    
    def __init__(self):
        self.engine = None
        self.session_factory = None
        
    async def setup(self):
        """Initialize async database engine and session factory."""
        logger.info("Setting up evaluation database session...")
        self.engine = create_async_engine(
            settings.DATABASE_ASYNC_URL,
            echo=False,
            future=True
        )
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        logger.success("Database session initialized")
        
    async def get_session(self) -> AsyncSession:
        """Get a new async database session."""
        return self.session_factory()
        
    async def cleanup(self):
        """Close database engine."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database engine disposed")


async def fetch_active_properties(
    session: AsyncSession,
    limit: int = 1000
) -> List[Property]:
    """
    Fetch all APPROVED properties from database.
    
    Args:
        session: Async database session
        limit: Maximum properties to fetch
        
    Returns:
        List of Property objects
    """
    logger.info(f"Fetching up to {limit} approved properties...")
    result = await session.execute(
        select(Property)
        .where(Property.status == PropertyStatus.APPROVED)
        .limit(limit)
    )
    properties = result.scalars().all()
    logger.success(f"Fetched {len(properties)} properties from database")
    return properties


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors (L2 normalized).
    
    Args:
        vec_a: First vector (normalized)
        vec_b: Second vector (normalized)
        
    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    # Both vectors are already L2 normalized, so similarity = dot product
    return float(np.dot(vec_a, vec_b))


def semantic_search_top_k(
    query_embedding: List[float],
    properties: List[Property],
    k: int = 10
) -> List[Tuple[Property, float]]:
    """
    Find top-k properties by cosine similarity to query embedding.
    Uses stored embeddings from database (JSON field).
    
    Args:
        query_embedding: Query embedding (384 dims, L2 normalized)
        properties: List of Property objects with embeddings
        k: Number of top results to return
        
    Returns:
        List of (Property, similarity_score) tuples, sorted by score descending
    """
    query_vec = np.array(query_embedding, dtype=np.float32)
    
    # Compute similarity scores for all properties
    scored_properties = []
    for prop in properties:
        if prop.embedding is None:
            continue
            
        prop_vec = np.array(prop.embedding, dtype=np.float32)
        
        # Ensure both vectors are L2 normalized
        query_vec_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        prop_vec_norm = prop_vec / (np.linalg.norm(prop_vec) + 1e-8)
        
        similarity = cosine_similarity(query_vec_norm, prop_vec_norm)
        scored_properties.append((prop, similarity))
    
    # Sort by similarity (descending) and return top-k
    scored_properties.sort(key=lambda x: x[1], reverse=True)
    return scored_properties[:k]


def get_ground_truth_relevance(
    property_id: UUID,
    title: str,
    description: str,
    price: float,
    location: str
) -> int:
    """
    Interactive prompt for human relevance annotation.
    
    Args:
        property_id: UUID of property
        title: Property title
        description: Property description
        price: Property price
        location: Property location
        
    Returns:
        Relevance score (0: Not relevant, 1: Somewhat relevant, 2: Relevant, 3: Highly relevant)
    """
    print("\n" + "="*80)
    print(f"Property ID: {property_id}")
    print(f"Title: {title}")
    print(f"Price: ₦{price:,.0f}")
    print(f"Location: {location}")
    print(f"Description: {description[:150]}..." if len(description) > 150 else f"Description: {description}")
    print("="*80)
    
    while True:
        try:
            score_str = input(
                "Enter relevance score (0=Not relevant, 1=Somewhat, 2=Relevant, 3=Highly relevant): "
            ).strip()
            score = int(score_str)
            if score in [0, 1, 2, 3]:
                return score
            else:
                print("❌ Invalid input. Please enter 0, 1, 2, or 3.")
        except ValueError:
            print("❌ Invalid input. Please enter a number (0-3).")


def calculate_ndcg_precision(
    relevance_scores: List[int],
    k_ndcg: int = 10,
    k_precision: int = 5
) -> Tuple[float, float]:
    """
    Calculate NDCG@k and Precision@k metrics.
    
    NDCG (Normalized Discounted Cumulative Gain):
        - Measures ranking quality considering position of relevant items
        - Relevance scores are discounted by log(position + 1)
        - Normalized against ideal ranking (sorted descending)
        - Range: 0.0 to 1.0 (1.0 = perfect ranking)
    
    Precision@k:
        - Proportion of top-k results that are relevant (score >= 2)
        - Range: 0.0 to 1.0
    
    Args:
        relevance_scores: List of relevance scores (0-3) in ranked order
        k_ndcg: NDCG cutoff (typically 10)
        k_precision: Precision cutoff (typically 5)
        
    Returns:
        Tuple of (NDCG@k, Precision@k)
    """
    relevance_array = np.array(relevance_scores[:k_ndcg], dtype=np.float32)
    
    # NDCG calculation using sklearn
    # Reshape for sklearn: predictions and true labels as 2D arrays
    y_true = np.array([relevance_array], dtype=np.float32)
    y_score = np.array([np.arange(len(relevance_array), 0, -1, dtype=np.float32)], dtype=np.float32)
    
    # Calculate ideal DCG (perfect ranking: sorted relevance descending)
    ideal_relevance = np.sort(relevance_array)[::-1]
    ideal_score = np.array([np.arange(len(ideal_relevance), 0, -1, dtype=np.float32)], dtype=np.float32)
    
    # DCG calculation: sum(rel_i / log2(i+1)) for i=1 to k
    positions = np.arange(1, len(relevance_array) + 1, dtype=np.float32)
    dcg = np.sum(relevance_array / np.log2(positions + 1))
    idcg = np.sum(ideal_relevance / np.log2(positions + 1))
    
    # NDCG = DCG / IDCG
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    # Precision@k: count of relevant items (score >= 2) in top-k
    relevant_at_k = np.sum(relevance_array[:k_precision] >= 2)
    precision = relevant_at_k / k_precision
    
    return float(ndcg), float(precision)


async def run_evaluation(eval_session: EvaluationSession):
    """
    Main evaluation workflow:
    1. Get test query from user
    2. Generate embedding
    3. Fetch and search properties
    4. Collect ground truth annotations
    5. Calculate and display metrics
    """
    session = await eval_session.get_session()
    
    try:
        # Step 1: Get test query
        print("\n" + "="*80)
        print("📊 SEMANTIC SEARCH EVALUATION: NDCG@10 & Precision@5")
        print("="*80)
        
        query = input("\nEnter your test query (e.g., '2-bedroom luxury apartment in Lekki'): ").strip()
        if not query:
            logger.error("Query cannot be empty")
            return
        
        # Step 2: Generate embedding
        logger.info(f"Generating embedding for query: '{query}'")
        query_embedding, embed_time_ms = generate_embedding(
            query,
            normalize=True,
            use_cache=False
        )
        logger.success(f"Query embedded in {embed_time_ms}ms ({EMBEDDING_DIMENSIONS} dimensions)")
        
        # Step 3: Fetch properties and search
        properties = await fetch_active_properties(session, limit=1000)
        if not properties:
            logger.error("No published properties found in database")
            return
        
        logger.info("Searching for top 10 results using cosine similarity...")
        top_results = semantic_search_top_k(query_embedding, properties, k=10)
        
        if not top_results:
            logger.error("No results found. Check that properties have embeddings.")
            return
        
        logger.success(f"Found {len(top_results)} results")
        
        # Step 4: Collect ground truth relevance scores
        print("\n" + "="*80)
        print("📝 GROUND TRUTH ANNOTATION")
        print("Review each result and assign a relevance score:")
        print("  0 = Not relevant (completely off-topic)")
        print("  1 = Somewhat relevant (tangentially related)")
        print("  2 = Relevant (matches query intent)")
        print("  3 = Highly relevant (excellent match)")
        print("="*80)
        
        relevance_scores = []
        for rank, (prop, similarity) in enumerate(top_results, 1):
            print(f"\n[Rank {rank}/10] Semantic Similarity: {similarity:.4f}")
            
            score = get_ground_truth_relevance(
                property_id=prop.id,
                title=prop.title,
                description=prop.description or "",
                price=prop.price,
                location=prop.location
            )
            relevance_scores.append(score)
        
        # Step 5: Calculate metrics
        print("\n" + "="*80)
        print("📈 EVALUATION RESULTS")
        print("="*80)
        
        ndcg_at_10, precision_at_5 = calculate_ndcg_precision(
            relevance_scores,
            k_ndcg=10,
            k_precision=5
        )
        
        # Calculate additional statistics
        relevant_count = sum(1 for s in relevance_scores if s >= 2)
        highly_relevant_count = sum(1 for s in relevance_scores if s >= 3)
        mean_relevance = np.mean(relevance_scores)
        
        # Display results
        print(f"\n✅ Query: '{query}'")
        print(f"✅ Results analyzed: {len(relevance_scores)}")
        print(f"\n📊 METRICS:")
        print(f"  • NDCG@10:          {ndcg_at_10:.4f}  (range 0-1, higher is better)")
        print(f"  • Precision@5:      {precision_at_5:.4f}  (range 0-1, higher is better)")
        print(f"\n📈 ADDITIONAL STATS:")
        print(f"  • Relevant items (score ≥ 2):     {relevant_count}/10 ({relevant_count*10}%)")
        print(f"  • Highly relevant (score ≥ 3):    {highly_relevant_count}/10 ({highly_relevant_count*10}%)")
        print(f"  • Mean relevance score:           {mean_relevance:.2f}/3.0")
        print(f"\n💾 RANKING DETAILS:")
        print(f"{'Rank':<6}{'Score':<8}{'Relevance':<12}{'Property Title':<40}")
        print("-" * 70)
        
        for rank, ((prop, sim), rel) in enumerate(zip(top_results, relevance_scores), 1):
            rel_label = {0: "Not Rel.", 1: "Somewhat", 2: "Relevant", 3: "Highly Rel."}[rel]
            title = prop.title[:36] + "..." if len(prop.title) > 36 else prop.title
            print(f"{rank:<6}{sim:<8.4f}{rel_label:<12}{title:<40}")
        
        print("\n" + "="*80)
        print("✨ Evaluation complete!")
        print("="*80)
        
    finally:
        await session.close()


async def main():
    """Entry point."""
    eval_session = EvaluationSession()
    await eval_session.setup()
    
    try:
        await run_evaluation(eval_session)
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    finally:
        await eval_session.cleanup()


if __name__ == "__main__":
    # Setup logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Run evaluation
    asyncio.run(main())
