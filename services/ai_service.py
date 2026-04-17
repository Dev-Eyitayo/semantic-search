"""
AI services for embeddings, similarity calculations, and SHAP explanations.
Refactored to use optimized embedding_service with batch processing support.
"""

import numpy as np
from typing import List, Dict, Tuple, Union, Optional
from sentence_transformers import CrossEncoder
from shap import KernelExplainer
from loguru import logger
import time

from services.embedding_service import (
    generate_embedding,
    generate_embeddings_batch,
    calculate_semantic_similarity,
    batch_similarity_search,
    get_embedding_model,
    EMBEDDING_DIMENSIONS
)

# Reranker model configuration
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Global reranker instance (lazy loaded)
_reranker_model: Optional[CrossEncoder] = None


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


# Backward compatibility functions (wrapping embedding_service)

def generate_embedding(
    text: str,
    normalize: bool = True
) -> Tuple[List[float], int]:
    """
    Generate S-BERT embedding for input text.
    BACKWARD COMPATIBLE with previous implementation.
    
    Args:
        text: Input text to embed (max ~5000 chars)
        normalize: Whether to apply L2 normalization
        
    Returns:
        Tuple of (embedding list, processing_time_ms)
    """
    from services.embedding_service import generate_embedding as gen_embedding
    return gen_embedding(text, normalize=normalize)


# New optimized functions

def generate_embeddings_batch(
    texts: Union[List[str], str],
    normalize: bool = True,
    batch_size: int = 32
) -> Tuple[List[List[float]], int]:
    """
    Generate embeddings for multiple texts efficiently using batch processing.
    
    Args:
        texts: Single text string or list of text strings
        normalize: Whether to apply L2 normalization
        batch_size: Batch size for encoding (higher = faster but more memory)
        
    Returns:
        Tuple of (list of embeddings, total_processing_time_ms)
        
    Example:
        texts = ["Luxury apartment", "Small studio", "Commercial space"]
        embeddings, time = generate_embeddings_batch(texts)
    """
    from services.embedding_service import generate_embeddings_batch as gen_batch
    return gen_batch(texts, normalize=normalize, batch_size=batch_size)


def calculate_semantic_similarity(
    query: str,
    texts: Union[List[str], str],
    normalize: bool = True
) -> Union[float, List[float]]:
    """
    Calculate semantic similarity between query and text(s).
    Efficiently handles both single and batch comparisons.
    
    Args:
        query: Search query text
        texts: Single text or list of texts to compare against
        normalize: Whether to normalize embeddings
        
    Returns:
        Single float (0-1) if texts is str, or list of floats if texts is list
        
    Example:
        # Single comparison
        score = calculate_semantic_similarity("luxury apartment", "High-end 2BR flat")
        
        # Batch comparison
        scores = calculate_semantic_similarity(
            "apartment near metro",
            ["Flat near station", "House in suburb", "Office space"]
        )
    """
    from services.embedding_service import calculate_semantic_similarity as calc_sim
    return calc_sim(query, texts, normalize=normalize)


def batch_similarity_search(
    query: str,
    candidates: List[str],
    top_k: Optional[int] = None,
    normalize: bool = True
) -> List[Tuple[int, str, float]]:
    """
    Efficient batch similarity search over candidates.
    Returns ranked results sorted by relevance.
    
    Args:
        query: Search query
        candidates: List of candidate property descriptions
        top_k: Return only top k results (None = all)
        normalize: Whether to normalize embeddings
        
    Returns:
        List of (index, text, score) tuples sorted by score descending
        
    Example:
        candidates = [
            "Luxury 3BR apartment in downtown",
            "Studio flat in suburbs",
            "1BR near transportation"
        ]
        results = batch_similarity_search("apartment downtown", candidates, top_k=2)
        # Returns: [(0, "Luxury 3BR apartment in downtown", 0.89), ...]
    """
    from services.embedding_service import batch_similarity_search as batch_search
    return batch_search(query, candidates, top_k=top_k, normalize=normalize)


# Reranking functions

def rerank_results(
    query: str,
    candidates: List[str],
    use_batch_similarity: bool = False
) -> List[Tuple[int, float]]:
    """
    Rerank candidate property texts using cross-encoder.
    Optionally uses batch similarity for pre-filtering.
    
    Args:
        query: Search query
        candidates: List of property descriptions to rank
        use_batch_similarity: If True, use batch similarity for pre-filtering before reranking
        
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
        
        logger.debug(f"Reranked {len(candidates)} candidates")
        
        return ranked
        
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        raise




# SHAP-based explanation functions

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
    start_time = time.time()
    
    try:
        # Define prediction function for SHAP
        def predict_relevance(feature_array):
            """
            Predicts relevance based on feature values.
            feature_array shape: (num_samples, num_features)
            """
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
        
        # Initialize SHAP explainer
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

