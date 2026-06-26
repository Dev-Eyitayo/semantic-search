#!/usr/bin/env python3
"""
Multi-Feature Ranking Ablation Study

This script demonstrates how your ranking formula's quality varies when removing features.
Tests 4 scenarios on the same query:
  A) Full Model:        Semantic: 0.55, Price: 0.20, Location: 0.15, Recency: 0.10
  B) No Price:          Semantic: 0.70, Price: 0.00, Location: 0.20, Recency: 0.10
  C) No Location:       Semantic: 0.65, Price: 0.25, Location: 0.00, Recency: 0.10
  D) Semantic Only:     Semantic: 1.00, Price: 0.00, Location: 0.00, Recency: 0.00

Shows how ranking order changes when features are ablated.

Usage:
    python eval_ablation_study.py
"""

import asyncio
import sys
import numpy as np
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timezone
from uuid import UUID
from dataclasses import dataclass

# Third-party imports
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.future import select
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
    batch_similarity_search
)


@dataclass
class WeightScenario:
    """Represents a ranking weight configuration."""
    name: str
    semantic: float
    price: float
    location: float
    recency: float
    description: str
    
    def validate(self):
        """Ensure weights sum to 1.0"""
        total = self.semantic + self.price + self.location + self.recency
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"{self.name}: weights sum to {total}, not 1.0")


class AblationSession:
    """Manages database session and ablation study context."""
    
    def __init__(self):
        self.engine = None
        self.session_factory = None
        
    async def setup(self):
        """Initialize async database engine and session factory."""
        logger.info("Setting up ablation study database session...")
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


def calculate_recency_score(created_at: datetime) -> float:
    """
    Calculate recency score based on listing age.
    Matches your production implementation.
    
    Args:
        created_at: Property creation timestamp (UTC)
        
    Returns:
        Recency score (0.0 to 1.0)
    """
    days_old = (datetime.now(timezone.utc) - created_at).days
    
    if days_old <= 7:
        return 0.95
    elif days_old <= 30:
        return 0.85
    elif days_old <= 90:
        return 0.70
    else:
        return 0.50


def calculate_price_score(query_price_max: Optional[float], actual_price: float) -> float:
    """
    Calculate price relevance score.
    Matches your production implementation.
    
    Args:
        query_price_max: Maximum price from user query filter
        actual_price: Property price
        
    Returns:
        Price score (0.0 to 1.0)
    """
    if query_price_max is None:
        return 0.5
    if actual_price <= query_price_max:
        return min(1.0, 1.0 - (actual_price / query_price_max * 0.5))
    return max(0.0, 1.0 - ((actual_price - query_price_max) / query_price_max))


def calculate_location_score_simple(
    query_location: Optional[str],
    actual_location: str
) -> float:
    """
    Calculate location relevance score.
    Simplified version for ablation study (matches production fallback).
    
    Args:
        query_location: Location from user query filter
        actual_location: Property location
        
    Returns:
        Location score (0.0 to 1.0)
    """
    if query_location is None:
        return 0.5
    if query_location.lower() in actual_location.lower():
        return 0.9
    return 0.3


def get_semantic_scores_batch(
    query: str,
    properties: List[Property]
) -> Dict[UUID, float]:
    """
    Calculate semantic similarity scores for all properties using batch search.
    
    Args:
        query: Search query
        properties: List of properties
        
    Returns:
        Dict mapping property ID to semantic score
    """
    logger.info(f"Computing semantic scores for {len(properties)} properties...")
    
    # Build candidate texts
    candidates = [
        f"{prop.title} {prop.description}" if prop.description else prop.title
        for prop in properties
    ]
    
    # Batch similarity search returns (index, text, score)
    results = batch_similarity_search(
        query=query,
        candidates=candidates,
        top_k=None,  # Get all results
        normalize=True
    )
    
    # Map back to property IDs
    semantic_scores = {}
    for idx, _, score in results:
        semantic_scores[properties[idx].id] = score
    
    # Fill in any missing with 0.0 (shouldn't happen)
    for prop in properties:
        if prop.id not in semantic_scores:
            semantic_scores[prop.id] = 0.0
    
    logger.success(f"Semantic scores computed for {len(semantic_scores)} properties")
    return semantic_scores


def calculate_final_score(
    semantic_score: float,
    price_score: float,
    location_score: float,
    recency_score: float,
    weights: WeightScenario
) -> float:
    """
    Calculate final ranking score using weighted combination.
    
    Args:
        semantic_score: Semantic relevance (0-1)
        price_score: Price alignment (0-1)
        location_score: Location proximity (0-1)
        recency_score: Listing recency (0-1)
        weights: Weight configuration
        
    Returns:
        Final ranking score (0-1)
    """
    return (
        semantic_score * weights.semantic +
        price_score * weights.price +
        location_score * weights.location +
        recency_score * weights.recency
    )


async def run_ablation_study(ablation_session: AblationSession):
    """
    Main ablation study workflow:
    1. Get query and optional filters from user
    2. Fetch and score properties
    3. Apply 4 weight scenarios
    4. Display ranking comparisons
    """
    session = await ablation_session.get_session()
    
    try:
        # Step 1: Get query and filters
        print("\n" + "="*100)
        print("🔬 MULTI-FEATURE RANKING ABLATION STUDY")
        print("="*100)
        print("\nCompares 4 weight scenarios on the same query:")
        print("  A) Full Model:      Semantic: 0.55, Price: 0.20, Location: 0.15, Recency: 0.10")
        print("  B) No Price:        Semantic: 0.70, Price: 0.00, Location: 0.20, Recency: 0.10")
        print("  C) No Location:     Semantic: 0.65, Price: 0.25, Location: 0.00, Recency: 0.10")
        print("  D) Semantic Only:   Semantic: 1.00, Price: 0.00, Location: 0.00, Recency: 0.00")
        print("="*100)
        
        query = input("\nEnter your test query (e.g., '2-bedroom apartment in Lekki'): ").strip()
        if not query:
            logger.error("Query cannot be empty")
            return
        
        # Optional filters
        price_max_input = input("Enter max price (optional, press Enter to skip): ").strip()
        price_max = float(price_max_input) if price_max_input else None
        
        location_filter = input("Enter location filter (optional, press Enter to skip): ").strip()
        location_filter = location_filter if location_filter else None
        
        # Step 2: Fetch properties and generate scores
        properties = await fetch_active_properties(session, limit=500)
        if not properties:
            logger.error("No published properties found in database")
            return
        
        # Generate semantic scores using batch search
        logger.info("Computing semantic similarity scores...")
        semantic_scores = get_semantic_scores_batch(query, properties)
        
        # Compute all feature scores for each property
        property_scores = {}
        logger.info("Computing feature scores for all properties...")
        for prop in properties:
            recency_score = calculate_recency_score(prop.created_at)
            price_score = calculate_price_score(price_max, prop.price)
            location_score = calculate_location_score_simple(location_filter, prop.location)
            semantic_score = semantic_scores.get(prop.id, 0.0)
            
            property_scores[prop.id] = {
                'property': prop,
                'semantic': semantic_score,
                'price': price_score,
                'location': location_score,
                'recency': recency_score
            }
        
        # Step 3: Define weight scenarios
        scenarios = [
            WeightScenario(
                name="A) Full Model",
                semantic=0.55,
                price=0.20,
                location=0.15,
                recency=0.10,
                description="All features balanced"
            ),
            WeightScenario(
                name="B) No Price",
                semantic=0.70,
                price=0.00,
                location=0.20,
                recency=0.10,
                description="Price feature removed"
            ),
            WeightScenario(
                name="C) No Location",
                semantic=0.65,
                price=0.25,
                location=0.00,
                recency=0.10,
                description="Location feature removed"
            ),
            WeightScenario(
                name="D) Semantic Only",
                semantic=1.00,
                price=0.00,
                location=0.00,
                recency=0.00,
                description="Only semantic similarity"
            ),
        ]
        
        # Validate scenarios
        for scenario in scenarios:
            scenario.validate()
        
        # Step 4: Calculate rankings for each scenario
        print("\n" + "="*100)
        print("📊 RANKING RESULTS BY SCENARIO")
        print("="*100)
        
        scenario_results = {}
        for scenario in scenarios:
            logger.info(f"Ranking for scenario: {scenario.name}")
            
            # Calculate final scores
            ranked_properties = []
            for prop_id, scores in property_scores.items():
                final_score = calculate_final_score(
                    semantic_score=scores['semantic'],
                    price_score=scores['price'],
                    location_score=scores['location'],
                    recency_score=scores['recency'],
                    weights=scenario
                )
                ranked_properties.append((scores['property'], final_score, scores))
            
            # Sort by final score (descending)
            ranked_properties.sort(key=lambda x: x[1], reverse=True)
            scenario_results[scenario.name] = (ranked_properties[:10], scenario)
        
        # Step 5: Display results side-by-side for top 10
        print(f"\n📋 Query: '{query}'")
        if price_max:
            print(f"   Price filter: ≤ ₦{price_max:,.0f}")
        if location_filter:
            print(f"   Location filter: {location_filter}")
        
        print("\n" + "="*100)
        print("TOP 10 PROPERTIES BY SCENARIO (Click ranking order)")
        print("="*100)
        
        # Create table headers
        max_title_len = 35
        for scenario_name, (ranked_props, scenario) in scenario_results.items():
            print(f"\n{'─' * 100}")
            print(f"{scenario_name:30s} | {scenario.description}")
            print(f"{'─' * 100}")
            print(f"{'Rank':<6}{'Final Score':<14}{'Semantic':<12}{'Price':<10}{'Location':<10}{'Recency':<10}{'Property Title':<40}")
            print(f"{'─' * 100}")
            
            for rank, (prop, final_score, feature_scores) in enumerate(ranked_props, 1):
                title = prop.title[:max_title_len]
                if len(prop.title) > max_title_len:
                    title += "..."
                
                print(
                    f"{rank:<6}"
                    f"{final_score:<14.4f}"
                    f"{feature_scores['semantic']:<12.4f}"
                    f"{feature_scores['price']:<10.4f}"
                    f"{feature_scores['location']:<10.4f}"
                    f"{feature_scores['recency']:<10.4f}"
                    f"{title:<40}"
                )
        
        # Step 6: Summary statistics
        print("\n" + "="*100)
        print("📈 RANKING STABILITY ANALYSIS")
        print("="*100)
        
        # Extract top-10 property IDs for each scenario
        full_model_ids = [prop.id for prop, _, _ in scenario_results["A) Full Model"][0]]
        
        print(f"\n{'Scenario':<30}{'Match with Full Model':<25}{'Interpretation'}")
        print("─" * 80)
        
        for scenario_name in ["B) No Price", "C) No Location", "D) Semantic Only"]:
            scenario_ids = [prop.id for prop, _, _ in scenario_results[scenario_name][0]]
            matches = len(set(full_model_ids) & set(scenario_ids))
            match_pct = (matches / 10) * 100
            
            interpretation = ""
            if match_pct >= 80:
                interpretation = "Very stable ✅"
            elif match_pct >= 60:
                interpretation = "Moderately stable ⚠️"
            else:
                interpretation = "Highly unstable ❌"
            
            print(f"{scenario_name:<30}{matches}/10 ({match_pct:.0f}%){'':<15}{interpretation}")
        
        print("\n" + "="*100)
        print("✨ Ablation study complete!")
        print("Insights:")
        print("  • High match % = Feature has minimal impact")
        print("  • Low match % = Feature significantly affects ranking")
        print("="*100)
        
    finally:
        await session.close()


async def main():
    """Entry point."""
    ablation_session = AblationSession()
    await ablation_session.setup()
    
    try:
        await run_ablation_study(ablation_session)
    except KeyboardInterrupt:
        print("\n\n⚠️  Ablation study interrupted by user")
    except Exception as e:
        logger.error(f"Ablation study failed: {e}")
        raise
    finally:
        await ablation_session.cleanup()


if __name__ == "__main__":
    # Setup logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Run ablation study
    asyncio.run(main())
