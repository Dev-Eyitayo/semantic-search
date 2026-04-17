"""
Optimized embedding service with singleton model loading and batch processing.
Uses all-MiniLM-L6-v2 for reduced model size and improved speed.
Thread-safe and process-safe with caching support.
"""

import time
import threading
from typing import List, Union, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from loguru import logger

# Model configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Smaller, faster model (384 dims vs 768)
EMBEDDING_DIMENSIONS = 384

# Global singleton instance with thread lock
_embedding_model: Optional[SentenceTransformer] = None
_model_lock = threading.Lock()

# Simple in-memory cache for embeddings (max 10k entries)
_embedding_cache: Dict[str, np.ndarray] = {}
_cache_lock = threading.Lock()
MAX_CACHE_SIZE = 10000


def _clear_cache_if_needed() -> None:
    """Clear cache if it exceeds max size (FIFO, clears half)."""
    with _cache_lock:
        if len(_embedding_cache) > MAX_CACHE_SIZE:
            # Remove oldest 50% to maintain size
            keys_to_remove = list(_embedding_cache.keys())[: MAX_CACHE_SIZE // 2]
            for key in keys_to_remove:
                del _embedding_cache[key]
            logger.warning(f"Embedding cache cleared (size exceeded {MAX_CACHE_SIZE})")


def get_embedding_model(warmup: bool = False) -> SentenceTransformer:
    """
    Get or initialize the embedding model (singleton with thread-safety).
    
    Args:
        warmup: If True, encodes a sample text to warm up the model
        
    Returns:
        SentenceTransformer instance (all-MiniLM-L6-v2)
        
    Raises:
        RuntimeError: If model fails to load
    """
    global _embedding_model
    
    if _embedding_model is not None:
        return _embedding_model
    
    with _model_lock:
        # Double-check after acquiring lock
        if _embedding_model is not None:
            return _embedding_model
        
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        try:
            _embedding_model = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                device="cuda" if _is_cuda_available() else "cpu"
            )
            logger.success(
                f"Embedding model loaded successfully: {EMBEDDING_MODEL_NAME} "
                f"({EMBEDDING_DIMENSIONS} dimensions)"
            )
            
            # Warmup the model if requested
            if warmup:
                logger.info("Warming up embedding model...")
                _warmup_model(_embedding_model)
                logger.success("Model warmup completed")
            
            return _embedding_model
        
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Embedding model initialization failed: {e}")


def _is_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _warmup_model(model: SentenceTransformer, num_samples: int = 3) -> None:
    """
    Warm up the embedding model by encoding sample texts.
    
    Args:
        model: SentenceTransformer instance
        num_samples: Number of sample texts to encode
    """
    warmup_texts = [
        "Luxury apartment with modern amenities in downtown area",
        "1 bedroom flat near transportation with beautiful view",
        "Commercial space available for rent in business district"
    ][:num_samples]
    
    try:
        model.encode(warmup_texts, show_progress_bar=False)
    except Exception as e:
        logger.warning(f"Model warmup failed (non-critical): {e}")


def generate_embedding(
    text: str,
    normalize: bool = True,
    use_cache: bool = True
) -> Tuple[List[float], int]:
    """
    Generate a single embedding for input text.
    
    Args:
        text: Input text to embed (max ~5000 chars)
        normalize: Whether to apply L2 normalization
        use_cache: Whether to use embedding cache
        
    Returns:
        Tuple of (embedding list, processing_time_ms)
        
    Raises:
        ValueError: If text is empty or exceeds length limit
    """
    if not text or len(text.strip()) == 0:
        raise ValueError("Text cannot be empty")
    
    if len(text) > 20000:  # ~5000 tokens at 4 chars per token
        raise ValueError("Text exceeds maximum length (~5000 tokens)")
    
    # Check cache
    if use_cache:
        cache_key = _get_cache_key(text)
        with _cache_lock:
            if cache_key in _embedding_cache:
                logger.debug(f"Cache hit for text (length: {len(text)})")
                cached_embedding = _embedding_cache[cache_key]
                embedding = cached_embedding.tolist()
                return embedding, 0  # Cache hit, no processing time
    
    start_time = time.time()
    
    try:
        model = get_embedding_model()
        
        # Generate embedding
        embedding_array = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        
        # L2 normalize if requested
        if normalize:
            embedding_array = _normalize_l2(embedding_array)
        
        # Convert to list
        embedding = embedding_array.tolist()
        
        # Cache the result
        if use_cache:
            _cache_embedding(text, embedding_array)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.debug(f"Embedding generated - Text length: {len(text)}, Time: {processing_time_ms}ms")
        
        return embedding, processing_time_ms
    
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise


def generate_embeddings_batch(
    texts: Union[List[str], str],
    normalize: bool = True,
    use_cache: bool = True,
    batch_size: int = 32
) -> Tuple[List[List[float]], int]:
    """
    Generate embeddings for multiple texts efficiently.
    
    Args:
        texts: Single text string or list of text strings
        normalize: Whether to apply L2 normalization
        use_cache: Whether to use embedding cache
        batch_size: Batch size for encoding (higher = faster but more memory)
        
    Returns:
        Tuple of (list of embeddings, total_processing_time_ms)
        
    Raises:
        ValueError: If texts is empty or invalid
    """
    # Convert single string to list
    if isinstance(texts, str):
        texts = [texts]
    
    if not texts:
        raise ValueError("Texts list cannot be empty")
    
    if not all(isinstance(t, str) for t in texts):
        raise ValueError("All items in texts must be strings")
    
    # Filter invalid texts
    valid_texts = [t for t in texts if t and len(t.strip()) > 0]
    if not valid_texts:
        raise ValueError("All texts are empty")
    
    start_time = time.time()
    embeddings_list: List[np.ndarray] = []
    cache_hits = 0
    cache_misses = 0
    texts_to_encode = []
    texts_to_encode_indices = []
    
    try:
        # Check cache for each text
        for idx, text in enumerate(valid_texts):
            if use_cache:
                cache_key = _get_cache_key(text)
                with _cache_lock:
                    if cache_key in _embedding_cache:
                        embeddings_list.append(_embedding_cache[cache_key])
                        cache_hits += 1
                        continue
            
            texts_to_encode.append(text)
            texts_to_encode_indices.append(idx)
            cache_misses += 1
        
        # Encode texts not in cache
        if texts_to_encode:
            model = get_embedding_model()
            
            # Batch encode
            encoded_arrays = model.encode(
                texts_to_encode,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Normalize if requested
            if normalize:
                encoded_arrays = np.array([_normalize_l2(arr) for arr in encoded_arrays])
            
            # Store in cache and result list
            for text, encoded_array in zip(texts_to_encode, encoded_arrays):
                if use_cache:
                    _cache_embedding(text, encoded_array)
                embeddings_list.append(encoded_array)
        
        # Convert to list format and maintain original order
        embeddings = [arr.tolist() for arr in embeddings_list]
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        logger.debug(
            f"Batch embeddings generated - Count: {len(embeddings)}, "
            f"Cache hits: {cache_hits}, Cache misses: {cache_misses}, "
            f"Time: {processing_time_ms}ms"
        )
        
        return embeddings, processing_time_ms
    
    except Exception as e:
        logger.error(f"Batch embedding generation failed: {e}")
        raise


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
        texts: Single text or list of texts to compare
        normalize: Whether to normalize (recommended for cosine similarity)
        
    Returns:
        Single float (0-1) if texts is str, or list of floats if texts is list
    """
    single_text = isinstance(texts, str)
    if single_text:
        texts = [texts]
    
    try:
        # Generate embeddings for all texts at once
        all_texts = [query] + texts
        embeddings, _ = generate_embeddings_batch(
            all_texts,
            normalize=normalize,
            batch_size=32
        )
        
        query_embedding = np.array(embeddings[0])
        text_embeddings = np.array(embeddings[1:])
        
        # Batch cosine similarity calculation
        if normalize:
            # L2 normalized: cosine similarity = dot product
            similarities = np.dot(text_embeddings, query_embedding)
        else:
            # Non-normalized: need explicit cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(
                text_embeddings,
                [query_embedding]
            ).flatten()
        
        # Ensure within [0, 1]
        similarities = np.clip(similarities, 0.0, 1.0)
        
        if single_text:
            return float(similarities[0])
        else:
            return similarities.tolist()
    
    except Exception as e:
        logger.error(f"Semantic similarity calculation failed: {e}")
        raise


def batch_similarity_search(
    query: str,
    candidates: List[str],
    top_k: Optional[int] = None,
    normalize: bool = True
) -> List[Tuple[int, str, float]]:
    """
    Efficient batch similarity search over candidates.
    Returns ranked results with indices, texts, and scores.
    
    Args:
        query: Search query
        candidates: List of candidate texts
        top_k: Return only top k results (None = all)
        normalize: Whether to normalize embeddings
        
    Returns:
        List of (index, text, score) tuples sorted by score descending
    """
    if not candidates:
        return []
    
    try:
        # Calculate similarities
        similarities = calculate_semantic_similarity(
            query,
            candidates,
            normalize=normalize
        )
        
        # Create tuples and sort
        ranked = [
            (idx, text, score)
            for idx, (text, score) in enumerate(zip(candidates, similarities))
        ]
        ranked.sort(key=lambda x: x[2], reverse=True)
        
        # Apply top_k filter
        if top_k:
            ranked = ranked[:top_k]
        
        return ranked
    
    except Exception as e:
        logger.error(f"Batch similarity search failed: {e}")
        raise


def _normalize_l2(vec: np.ndarray) -> np.ndarray:
    """L2 normalize a vector."""
    magnitude = np.linalg.norm(vec)
    if magnitude > 0:
        return vec / magnitude
    return vec


def _get_cache_key(text: str) -> str:
    """
    Generate cache key from text.
    Uses hash for efficiency (collision risk is negligible for our use case).
    """
    # Simple hash-based key (fast, small collision risk acceptable here)
    return f"emb_{hash(text) & 0x7fffffff}"


def _cache_embedding(text: str, embedding: np.ndarray) -> None:
    """Store embedding in cache."""
    with _cache_lock:
        cache_key = _get_cache_key(text)
        _embedding_cache[cache_key] = embedding
        
        # Trigger cache cleanup if needed
        if len(_embedding_cache) > MAX_CACHE_SIZE * 1.1:
            _clear_cache_if_needed()


def clear_embedding_cache() -> int:
    """
    Clear the embedding cache.
    
    Returns:
        Number of entries cleared
    """
    with _cache_lock:
        count = len(_embedding_cache)
        _embedding_cache.clear()
        logger.info(f"Embedding cache cleared ({count} entries removed)")
        return count


def get_cache_stats() -> Dict[str, int]:
    """Get current cache statistics."""
    with _cache_lock:
        return {
            "cache_size": len(_embedding_cache),
            "max_size": MAX_CACHE_SIZE,
            "usage_percent": round((len(_embedding_cache) / MAX_CACHE_SIZE) * 100, 2)
        }


def reset_embedding_model() -> None:
    """
    Reset the embedding model (useful for testing or emergency cleanup).
    Thread-safe.
    """
    global _embedding_model
    
    with _model_lock:
        if _embedding_model is not None:
            logger.warning("Resetting embedding model")
            _embedding_model = None
