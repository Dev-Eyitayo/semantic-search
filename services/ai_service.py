"""
AI services for embeddings and SHAP explanations.
Uses sentence-transformers for embeddings and SHAP for feature attribution.
"""

import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from shap import KernelExplainer
import logging
from loguru import logger

# Model configurations
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Global model instances (lazy loaded)
_embedding_model = None
_reranker_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Get or initialize the embedding model (lazy loading).
    Model is loaded once and reused for all requests.
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        try:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.success(f"Embedding model loaded successfully: {EMBEDDING_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Embedding model initialization failed: {e}")
    return _embedding_model


def get_reranker_model() -> CrossEncoder:
    """
    Get or initialize the cross-encoder reranker model (lazy loading).
    Model is loaded once and reused for all requests.
    """
    global _reranker_model
    if _reranker_model is None:
        logger.info(f"Loading reranker model: {RERANKER_MODEL_NAME}")
        try:
            _reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
            logger.success(f"Reranker model loaded successfully: {RERANKER_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise RuntimeError(f"Reranker model initialization failed: {e}")
    return _reranker_model


def generate_embedding(text: str, normalize: bool = False) -> Tuple[List[float], int]:
    """
    Generate S-BERT embedding for input text.
    
    Args:
        text: Input text to embed (max ~5000 chars)
        normalize: Whether to apply L2 normalization
        
    Returns:
        Tuple of (embedding list, processing_time_ms)
    """
    import time
    
    if not text or len(text.strip()) == 0:
        raise ValueError("Text cannot be empty")
    
    # Rough token limit check (approx 4 chars per token)
    if len(text) > 20000:  # ~5000 tokens at 4 chars per token
        raise ValueError("Text exceeds maximum length (~5000 tokens)")
    
    start_time = time.time()
    
    try:
        model = get_embedding_model()
        
        # Generate embedding
        embedding_array = model.encode(text, convert_to_numpy=True)
        
        # Convert to list
        embedding = embedding_array.tolist()
        
        # L2 normalize if requested (for cosine similarity)
        if normalize:
            magnitude = np.sqrt(np.sum(np.array(embedding) ** 2))
            if magnitude > 0:
                embedding = (np.array(embedding) / magnitude).tolist()
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        logger.debug(f"Embedding generated - Text length: {len(text)}, Time: {processing_time_ms}ms")
        
        return embedding, processing_time_ms
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise


def calculate_semantic_similarity(query: str, property_text: str) -> float:
    """
    Calculate semantic similarity between a query and property text using S-BERT.
    
    Args:
        query: Search query
        property_text: Property title + description
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    try:
        model = get_embedding_model()
        
        # Encode both texts
        query_embedding = model.encode(query, convert_to_numpy=True)
        property_embedding = model.encode(property_text, convert_to_numpy=True)
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(
            [query_embedding],
            [property_embedding]
        )[0][0]
        
        # Ensure within [0, 1]
        similarity = float(similarity)
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity
        
    except Exception as e:
        logger.error(f"Semantic similarity calculation failed: {e}")
        raise


def rerank_results(query: str, candidates: List[str]) -> List[Tuple[int, float]]:
    """
    Rerank candidate property texts using cross-encoder.
    
    Args:
        query: Search query
        candidates: List of property descriptions to rank
        
    Returns:
        List of (index, score) tuples sorted by score descending
    """
    if not candidates:
        return []
    
    try:
        model = get_reranker_model()
        
        # Create query-candidate pairs
        pairs = [[query, candidate] for candidate in candidates]
        
        # Score all pairs
        scores = model.predict(pairs)
        
        # Return sorted by score
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked
        
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        raise


def compute_shap_explanation(
    query: str,
    property_data: Dict,
    feature_values: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute SHAP values for feature attribution.
    Uses KernelExplainer to estimate feature importance.
    
    Args:
        query: Search query
        property_data: Property metadata (for context)
        feature_values: Dictionary of feature names to their scores
                       e.g., {"semantic_score": 0.83, "price_score": 0.91, ...}
        
    Returns:
        Dictionary of feature names to Shapley values
    """
    import time
    start_time = time.time()
    
    try:
        # Define prediction function for SHAP
        def predict_relevance(feature_array):
            """
            Predicts relevance based on feature values.
            feature_array shape: (num_samples, num_features)
            """
            # Weighted sum of features
            if len(feature_array.shape) == 1:
                feature_array = feature_array.reshape(1, -1)
            
            # Default weights (can be overridden by ranking config)
            weights = np.array([0.55, 0.20, 0.15, 0.10])  # semantic, price, location, recency
            
            # Ensure weights match features
            if feature_array.shape[1] > len(weights):
                weights = np.concatenate([weights, np.zeros(feature_array.shape[1] - len(weights))])
            else:
                weights = weights[:feature_array.shape[1]]
            
            # Weighted sum
            predictions = np.dot(feature_array, weights)
            return predictions
        
        # Get feature values as array
        feature_names = list(feature_values.keys())
        feature_array = np.array(list(feature_values.values())).reshape(1, -1)
        
        # Use baseline (average expected values)
        background = np.array([
            [0.5] * len(feature_names)  # Neutral baseline
        ])
        
        # Initialize SHAP explainer (KernelExplainer is model-agnostic)
        explainer = KernelExplainer(
            model=predict_relevance,
            data=background,
            link="identity"
        )
        
        # Compute SHAP values
        shap_values = explainer.shap_values(feature_array)
        
        # Extract and normalize
        shap_dict = {}
        if len(shap_values) > 0:
            shap_array = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            for name, value in zip(feature_names, shap_array):
                shap_dict[name] = float(value)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.debug(f"SHAP explanation computed in {processing_time_ms}ms")
        
        return shap_dict
        
    except Exception as e:
        logger.error(f"SHAP explanation computation failed: {e}")
        # Return equal importance if SHAP fails
        num_features = len(feature_values)
        if num_features > 0:
            equal_importance = 1.0 / num_features
            return {name: equal_importance for name in feature_values.keys()}
        return {}


def get_feature_human_labels(feature_name: str, feature_value: float) -> Tuple[str, str]:
    """
    Convert feature names and values to human-readable labels.
    
    Args:
        feature_name: Name of the feature (e.g., "semantic_score")
        feature_value: Score value (0.0-1.0)
        
    Returns:
        Tuple of (human_label, direction)
    """
    label_map = {
        "semantic_score": {
            "high": "Strongly matches your search intent",
            "medium": "Partially matches your search intent",
            "low": "Weakly matches your search intent"
        },
        "price_score": {
            "high": "Well within your budget",
            "medium": "Matches your budget range",
            "low": "Above your budget"
        },
        "location_score": {
            "high": "Located in your preferred area",
            "medium": "Located near your preferred area",
            "low": "Different location than preferred"
        },
        "recency_score": {
            "high": "Recently listed — likely still available",
            "medium": "Moderately recent listing",
            "low": "Older listing"
        },
        "bedroom_match": {
            "high": "Exact bedroom count match",
            "medium": "Close bedroom count match",
            "low": "Different bedroom count"
        },
        "amenity_security": {
            "high": "Has the security features you need",
            "medium": "Has some security features",
            "low": "Missing security features"
        },
        "property_type_match": {
            "high": "Matches your property type preference",
            "medium": "Similar to preferred property type",
            "low": "Different property type"
        }
    }
    
    # Determine intensity level
    if feature_value >= 0.75:
        intensity = "high"
    elif feature_value >= 0.5:
        intensity = "medium"
    else:
        intensity = "low"
    
    # Get label
    if feature_name in label_map:
        label = label_map[feature_name].get(intensity, f"{feature_name}")
    else:
        label = f"{feature_name}: {feature_value:.2f}"
    
    # Determine direction
    direction = "positive" if feature_value > 0.5 else ("neutral" if feature_value == 0.5 else "negative")
    
    return label, direction
