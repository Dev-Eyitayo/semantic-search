from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, and_, or_, desc, insert, text
from datetime import datetime, timezone
import uuid
import time
import re
from typing import Optional, List

from api.deps import get_current_user
from db.session import get_db
from db.models.user import User
from db.models.property import Property
from db.models.search import SearchLog, SearchFeedback, RankingConfig
from core.enums import PropertyStatus
from schemas.search import (
    SemanticSearchRequest, SemanticSearchResponse, SearchResult,
    KeywordSearchResponse, SuggestionsResponse, SimilarPropertiesResponse,
    SearchFeedbackRequest, SearchFeedbackResponse, ExplanationFeature
)
from schemas.base import StandardResponse
from services.redis_service import redis_client
from services.embedding_service import batch_similarity_search
from loguru import logger
import json

router = APIRouter()


def calculate_semantic_score(query: str, description: str, title: str) -> float:
    """
    Calculate semantic similarity score between query and property.
    Placeholder for S-BERT embedding similarity.
    """
    # This would use sentence-transformers/all-mpnet-base-v2 in production
    query_lower = query.lower()
    text = f"{title} {description}".lower()
    
    # Simple keyword matching as fallback
    keywords = query_lower.split()
    matches = sum(1 for kw in keywords if kw in text)
    return min(0.99, matches / len(keywords) if keywords else 0.5)


def calculate_price_score(query_price_max: Optional[float], actual_price: float) -> float:
    """Calculate price relevance score"""
    if query_price_max is None:
        return 0.5
    if actual_price <= query_price_max:
        return min(1.0, 1.0 - (actual_price / query_price_max * 0.5))
    return max(0.0, 1.0 - ((actual_price - query_price_max) / query_price_max))


def calculate_location_score(query_location: Optional[str], actual_location: str) -> float:
    """Calculate location relevance score"""
    if query_location is None:
        return 0.5
    if query_location.lower() in actual_location.lower():
        return 0.9
    return 0.3


def extract_locations_from_query(query: str, all_locations: list[str]) -> list[str]:
    """
    Extract location names mentioned in the query text.
    Matches against actual property locations with flexible matching.
    """
    query_lower = query.lower()
    found_locations = []
    
    # Match against all known locations with flexible matching
    for location in all_locations:
        location_lower = location.lower()
        
        # Exact match or partial match
        if location_lower in query_lower:
            found_locations.append(location)
        # Also check for partial word matches (e.g., "lekki" in "lekki phase 1")
        elif any(part in query_lower for part in location_lower.split(',')):
            found_locations.append(location)
    
    return list(set(found_locations))  # Remove duplicates


def extract_property_type_from_query(query: str) -> Optional[str]:
    """
    Extract property type from query.
    Returns: apartment, house, duplex, studio, or None
    """
    query_lower = query.lower()
    property_types = {
        'apartment': ['apartment', 'apt', 'flat', 'condo'],
        'house': ['house', 'home', 'residential', 'bungalow'],
        'duplex': ['duplex', 'maisonette'],
        'studio': ['studio', 'bedsitter', 'bedsit', 'single room']
    }
    
    for prop_type, keywords in property_types.items():
        if any(keyword in query_lower for keyword in keywords):
            return prop_type
    
    return None


def extract_amenities_from_query(query: str) -> list[str]:
    """
    Extract amenity keywords from query.
    Returns list of amenities mentioned.
    """
    query_lower = query.lower()
    amenity_keywords = {
        'swimming pool': ['pool', 'swimming', 'swim'],
        'gym': ['gym', 'fitness', 'workout'],
        'garden': ['garden', 'lawn', 'landscap', 'green space'],
        'parking': ['parking', 'garage', 'carport'],
        'security': ['secure', 'security', 'gated', 'guard', '24/7'],
        'air conditioning': ['ac', 'air condition'],
        'backup generator': ['generator', 'backup power'],
        'water supply': ['water', 'borehol'],
        'elevator': ['elevator', 'lift', 'highrise'],
        'balcony': ['balcony', 'terrace', 'patio'],
        'kitchen': ['kitchen', 'cook'],
    }
    
    found_amenities = []
    for amenity, keywords in amenity_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            found_amenities.append(amenity)
    
    return found_amenities


def extract_budget_from_query(query: str) -> Optional[float]:
    """
    Extract budget/price mentions from query.
    Looks for patterns like "2.5M", "2.5 million", "budget 2500000", etc.
    """
    import re
    query_lower = query.lower()
    
    # Pattern: number + M/million/thousand
    patterns = [
        r'(\d+(?:\.\d+)?)\s*m(?:illion)?',  # 2.5m or 2.5 million
        r'budget\s*[₦:]*\s*(\d+(?:\.\d+)?)',  # budget 2.5
        r'([๦₦]*\s*\d+(?:\.\d+)?)(?:\s*m|million)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            try:
                amount = float(match.group(1))
                # Convert to Naira if in millions
                if 'm' in match.group(0).lower() or 'million' in match.group(0).lower():
                    amount *= 1_000_000
                return amount
            except (ValueError, IndexError):
                continue
    
    return None


def extract_bedrooms_from_query(query: str) -> Optional[int]:
    """
    Extract number of bedrooms from query.
    Looks for patterns like "1 bedroom", "2-bedroom", "3br", etc.
    """
    import re
    query_lower = query.lower()
    
    patterns = [
        r'(\d+)\s*-?bedroom',
        r'(\d+)\s*br(?:ooms?)?',
        r'(\d+)\s*bed',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue
    
    return None


def calculate_location_score_enhanced(
    query_mentioned_locations: list[str],
    filter_location: Optional[str],
    actual_location: str
) -> float:
    """
    Enhanced location scoring that considers:
    1. Explicit filter location (highest priority)
    2. Location mentions in semantic query text
    3. Default score if no location mentioned
    """
    # Explicit filter location has highest priority
    if filter_location and filter_location.lower() in actual_location.lower():
        return 0.95
    
    # Check if any query-mentioned locations match
    if query_mentioned_locations:
        for mentioned_loc in query_mentioned_locations:
            if mentioned_loc.lower() in actual_location.lower():
                return 0.85  # High score for query-mentioned location
    
    # Partial matches get moderate score
    if filter_location and (filter_location.lower() in actual_location.lower() or 
                           actual_location.lower() in filter_location.lower()):
        return 0.7
    
    # Default score when no location specified or no match
    return 0.5


async def get_dynamic_weights(db: AsyncSession, query: str) -> dict:
    """
    Get ranking weights and adjust them based on query content.
    If location is explicitly mentioned in query, increase location weight.
    """
    result = await db.execute(
        select(RankingConfig).order_by(RankingConfig.updated_at.desc()).limit(1)
    )
    config = result.scalars().first()
    
    if config:
        weights = {
            'semantic_score': config.semantic_score_weight,
            'price_score': config.price_score_weight,
            'location_score': config.location_score_weight,
            'recency_score': config.recency_score_weight,
        }
    else:
        weights = {
            'semantic_score': 0.50,
            'price_score': 0.20,
            'location_score': 0.20,
            'recency_score': 0.10,
        }
    
    # Boost location weight if location keywords detected in query
    location_keywords = ['in', 'lekki', 'lagos', 'abuja', 'yaba', 'ikeja', 'vi', 
                        'ikoyi', 'surulere', 'ajah', 'maryland', 'magodo', 'ketu',
                        'calabar', 'benin', 'portharcourt', 'ibadan', 'enugu', 'kano',
                        'kaduna', 'jos', 'bauchi', 'wuse', 'maitama', 'garki', 'asokoro']
    
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in location_keywords):
        # If location is mentioned, increase its weight
        weights['location_score'] = min(0.35, weights['location_score'] + 0.10)
        weights['semantic_score'] = max(0.35, weights['semantic_score'] - 0.10)
        logger.debug("Location boost applied: detected location keywords in query")
    
    return weights


def calculate_recency_score(created_at: datetime) -> float:
    """Calculate recency score based on listing age"""
    days_old = (datetime.now(timezone.utc) - created_at).days
    if days_old <= 7:
        return 0.95
    elif days_old <= 30:
        return 0.85
    elif days_old <= 90:
        return 0.70
    else:
        return 0.50


def calculate_final_ranking_score(semantic: float, price: float, location: float, recency: float, weights: dict) -> float:
    """Calculate final weighted ranking score"""
    return (
        semantic * weights.get('semantic_score', 0.55) +
        price * weights.get('price_score', 0.20) +
        location * weights.get('location_score', 0.15) +
        recency * weights.get('recency_score', 0.10)
    )




async def log_search_async(
    db: AsyncSession,
    user_id: Optional[uuid.UUID],
    query: str,
    filters: Optional[dict],
    results_count: int,
    search_type: str,
    processing_time_ms: int
):
    """Log search query asynchronously"""
    try:
        log_entry = SearchLog(
            user_id=user_id,
            query=query,
            filters=filters,
            results_count=results_count,
            search_type=search_type,
            processing_time_ms=processing_time_ms,
            embedding_model="all-mpnet-base-v2"
        )
        db.add(log_entry)
        await db.commit()
        logger.debug(f"Search logged: {query} ({search_type})")
    except Exception as e:
        logger.error(f"Error logging search: {e}")


@router.post("/semantic", response_model=StandardResponse[SemanticSearchResponse], tags=["Search"])
async def semantic_search(
    request: SemanticSearchRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[User] = Depends(lambda: None),  # Optional auth
    db: AsyncSession = Depends(get_db)
):
    """
    Smart semantic search with intelligent query parsing.
    Extracts: locations, property types, amenities, bedrooms, budget from query text.
    """
    query_start = time.time()
    logger.info(f"Semantic search initiated - Query: {request.query}, Explain: {request.explain}")
    
    # Validate query
    if not request.query or len(request.query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # SMART EXTRACTION: Parse query for structured attributes
    extracted_bedrooms = extract_bedrooms_from_query(request.query)
    extracted_property_type = extract_property_type_from_query(request.query)
    extracted_amenities = extract_amenities_from_query(request.query)
    extracted_budget = extract_budget_from_query(request.query)
    
    logger.debug(f"Query extraction - Bedrooms: {extracted_bedrooms}, Type: {extracted_property_type}, "
                f"Amenities: {extracted_amenities}, Budget: {extracted_budget}")
    
    # Build filter conditions
    filters = [Property.status == PropertyStatus.APPROVED]
    
    # Apply explicit filters first
    if request.filters:
        if request.filters.get('price_min'):
            filters.append(Property.price >= request.filters['price_min'])
        if request.filters.get('price_max'):
            filters.append(Property.price <= request.filters['price_max'])
        if request.filters.get('location'):
            filters.append(Property.location.ilike(f"%{request.filters['location']}%"))
        if request.filters.get('bedrooms'):
            filters.append(Property.bedrooms == request.filters['bedrooms'])
        if request.filters.get('bathrooms'):
            filters.append(Property.bathrooms == request.filters['bathrooms'])
        if request.filters.get('property_type'):
            filters.append(Property.property_type == request.filters.get('property_type'))
    
    # SMART FILTERS: Apply extracted attributes if not already in explicit filters
    if extracted_bedrooms and not request.filters.get('bedrooms'):
        filters.append(Property.bedrooms == extracted_bedrooms)
        logger.debug(f"Applied extracted bedroom filter: {extracted_bedrooms}")
    
    if extracted_property_type and not request.filters.get('property_type'):
        filters.append(Property.property_type == extracted_property_type)
        logger.debug(f"Applied extracted property type filter: {extracted_property_type}")
    
    if extracted_budget and not request.filters.get('price_max'):
        filters.append(Property.price <= extracted_budget)
        logger.debug(f"Applied extracted budget filter: ₦{extracted_budget:,.0f}")
    
    # Get dynamic ranking weights (adjusted based on query content)
    weights = await get_dynamic_weights(db, request.query)
    
    # Fetch all matching properties
    result = await db.execute(
        select(Property).where(and_(*filters)).limit(1000)  # Batch processing limit
    )
    properties = result.scalars().all()
    
    if not properties:
        return SemanticSearchResponse(
            results=[],
            total=0,
            processing_time_ms=int((time.time() - query_start) * 1000)
        )
    
    # Extract locations mentioned in the semantic query for enhanced scoring
    all_unique_locations = list(set(p.location for p in properties if p.location))
    query_mentioned_locations = extract_locations_from_query(request.query, all_unique_locations)
    
    # Use BATCH similarity search for 10-50x performance improvement!
    candidates = [
        f"{prop.title}. {prop.description}" for prop in properties
    ]
    
    # Batch search returns top results efficiently
    batch_start = time.time()
    batch_results = batch_similarity_search(
        query=request.query,
        candidates=candidates,
        top_k=min(len(candidates), 100),  # Get top 100 results
        normalize=True
    )
    batch_time_ms = int((time.time() - batch_start) * 1000)
    
    logger.debug(f"Batch similarity search: {len(batch_results)} results in {batch_time_ms}ms")
    
    # Calculate scores for each result from batch
    scored_properties = []
    for idx, text, batch_semantic_score in batch_results:
        prop = properties[idx]
        
        semantic_score = batch_semantic_score
        price_score = calculate_price_score(
            request.filters.get('price_max') if request.filters else None,
            prop.price
        )
        # Use enhanced location scoring with query-extracted locations
        location_score = calculate_location_score_enhanced(
            query_mentioned_locations=query_mentioned_locations,
            filter_location=request.filters.get('location') if request.filters else None,
            actual_location=prop.location
        )
        recency_score = calculate_recency_score(prop.created_at)
        
        final_score = calculate_final_ranking_score(
            semantic_score, price_score, location_score, recency_score, weights
        )
        
        # ATTRIBUTE MATCHING BONUSES: Boost score if property matches extracted attributes
        attribute_match_bonus = 0.0
        attribute_matches = []
        
        # Bonus for property type match
        if extracted_property_type and str(prop.property_type.value).lower() == extracted_property_type.lower():
            attribute_match_bonus += 0.08
            attribute_matches.append(f"Type: {extracted_property_type}")
        
        # Bonus for bedroom count match
        if extracted_bedrooms and prop.bedrooms == extracted_bedrooms:
            attribute_match_bonus += 0.05
            attribute_matches.append(f"Bedrooms: {extracted_bedrooms}")
        
        # Bonus for amenities Match
        if extracted_amenities and prop.amenities:
            matching_amenities = sum(1 for amenity in extracted_amenities 
                                    if any(amenity.lower() in a.lower() for a in prop.amenities))
            if matching_amenities > 0:
                amenity_bonus = min(0.08, matching_amenities * 0.04)  # Up to 8% for all amenities
                attribute_match_bonus += amenity_bonus
                attribute_matches.append(f"Amenities: {matching_amenities} match")
        
        # Apply bonus to final score (cap at 0.95 to preserve realism)
        final_score_boosted = min(0.95, final_score + attribute_match_bonus)
        
        # Track attribute matches for explanations
        match_info = {
            'boost': attribute_match_bonus,
            'matches': attribute_matches
        }
        
        # Generate explanations if requested
        explanations = None
        explanation_summary = None
        
        if request.explain:
            explanations = [
                ExplanationFeature(
                    feature="semantic_score",
                    label="Strongly matches your search intent" if semantic_score > 0.7 else "Partially matches your search",
                    weight=semantic_score,
                    direction="positive" if semantic_score > 0.5 else "neutral"
                ),
                ExplanationFeature(
                    feature="price_score",
                    label="Matches your budget" if price_score > 0.7 else "Above your budget",
                    weight=price_score,
                    direction="positive" if price_score > 0.5 else "negative"
                ),
                ExplanationFeature(
                    feature="location_score",
                    label="Located in your preferred area" if location_score > 0.9 else ("Partially matches your location" if location_score > 0.6 else "Different location"),
                    weight=location_score,
                    direction="positive" if location_score > 0.5 else "neutral"
                ),
                ExplanationFeature(
                    feature="recency_score",
                    label="Recently listed — likely still available" if recency_score > 0.8 else "Older listing",
                    weight=recency_score,
                    direction="positive" if recency_score > 0.7 else "neutral"
                )
            ]
            
            # Build explanation summary including attribute matches
            match_text = " + ".join(match_info['matches']) if match_info['matches'] else ""
            if match_text:
                explanation_summary = f"Matches: {match_text}. Strongly relevant!" if final_score_boosted > 0.75 else f"Matches: {match_text}"
            else:
                explanation_summary = "Strongly matches your search criteria" if final_score_boosted > 0.75 else "Moderately relevant to your search"
        
        scored_properties.append({
            'property': prop,
            'semantic_score': semantic_score,
            'price_score': price_score,
            'location_score': location_score,
            'recency_score': recency_score,
            'final_score': final_score_boosted,  # Use boosted score
            'explanations': explanations,
            'explanation_summary': explanation_summary
        })
    
    # Batch results already sorted by relevance - no need to sort again!
    total_results = len(scored_properties)
    offset = (request.page - 1) * request.limit
    paginated = scored_properties[offset:offset + request.limit]
    
    # Build response
    results = []
    for item in paginated:
        prop = item['property']
        results.append(SearchResult(
            id=prop.id,
            title=prop.title,
            price=prop.price,
            price_type=prop.price_type,
            location=prop.location,
            bedrooms=prop.bedrooms,
            bathrooms=prop.bathrooms,
            thumbnail=prop.thumbnail,
            ranking_score=item['final_score'],
            semantic_score=item['semantic_score'],
            price_score=item['price_score'],
            location_score=item['location_score'],
            recency_score=item['recency_score'],
            explanations=item['explanations'],
            explanation_summary=item['explanation_summary']
        ))
    
    processing_time_ms = int((time.time() - query_start) * 1000)
    
    # Log search asynchronously
    background_tasks.add_task(
        log_search_async,
        db,
        current_user.id if current_user else None,
        request.query,
        request.filters,
        total_results,
        "semantic",
        processing_time_ms
    )
    
    logger.success(f"Semantic search completed - Results: {len(results)}, Time: {processing_time_ms}ms")
    
    return StandardResponse(
        message="Search results retrieved successfully",
        data=SemanticSearchResponse(
            query=request.query,
            total_results=total_results,
            page=request.page,
            limit=request.limit,
            processing_time_ms=processing_time_ms,
            results=results
        )
    )


@router.get("/keyword", response_model=StandardResponse[KeywordSearchResponse], tags=["Search"])
async def keyword_search(
    q: str = Query(..., min_length=1, max_length=500),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50),
    location: Optional[str] = Query(None),
    bedrooms: Optional[int] = Query(None),
    price_max: Optional[float] = Query(None, ge=0),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: Optional[User] = Depends(lambda: None),
    db: AsyncSession = Depends(get_db)
):
    """
    Lightweight full-text keyword search using PostgreSQL tsvector.
    Fallback when semantic search is unavailable.
    """
    query_start = time.time()
    logger.info(f"Keyword search initiated - Query: {q}")
    
    # Build filter conditions
    filters = [Property.status == PropertyStatus.APPROVED]
    
    # Full-text search on title and description
    search_vector = func.to_tsvector('english', Property.title + ' ' + Property.description)
    query_tsquery = func.plainto_tsquery('english', q)
    filters.append(search_vector.match(query_tsquery))
    
    if location:
        filters.append(Property.location.ilike(f"%{location}%"))
    if bedrooms:
        filters.append(Property.bedrooms == bedrooms)
    if price_max:
        filters.append(Property.price <= price_max)
    
    # Get total count
    count_result = await db.execute(
        select(func.count(Property.id)).where(and_(*filters))
    )
    total = count_result.scalar()
    
    # Get paginated results
    offset = (page - 1) * limit
    result = await db.execute(
        select(Property)
        .where(and_(*filters))
        .order_by(Property.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    properties = result.scalars().all()
    
    # Build response
    results = [
        SearchResult(
            id=prop.id,
            title=prop.title,
            price=prop.price,
            price_type=prop.price_type,
            location=prop.location,
            bedrooms=prop.bedrooms,
            bathrooms=prop.bathrooms,
            thumbnail=prop.thumbnail,
            ranking_score=0.5,
            semantic_score=0.5,
            price_score=0.5,
            location_score=0.5
        )
        for prop in properties
    ]
    
    processing_time_ms = int((time.time() - query_start) * 1000)
    
    # Log search asynchronously
    background_tasks.add_task(
        log_search_async,
        db,
        current_user.id if current_user else None,
        q,
        {"location": location, "bedrooms": bedrooms, "price_max": price_max},
        total,
        "keyword",
        processing_time_ms
    )
    
    logger.success(f"Keyword search completed - Results: {len(results)}")
    
    return StandardResponse(
        message="Keyword search results retrieved successfully",
        data=KeywordSearchResponse(
            query=q,
            results=results,
            total=total,
            page=page,
            search_type="keyword_fallback"
        )
    )


@router.get("/suggestions", response_model=StandardResponse[SuggestionsResponse], tags=["Search"])
async def get_search_suggestions(
    q: str = Query(..., min_length=2, max_length=500),
    limit: int = Query(5, ge=1, le=10),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns query autocompletion suggestions based on popular past queries and locations.
    """
    logger.info(f"Suggestions requested for: {q}")
    
    # Try to get from Redis cache first
    cache_key = f"search_suggestions:{q.lower()}"
    try:
        cached_suggestions = await redis_client.get(cache_key)
        if cached_suggestions:
            suggestions = json.loads(cached_suggestions)
            logger.debug(f"Suggestions retrieved from cache")
            return StandardResponse(
                message="Suggestions retrieved successfully",
                data=SuggestionsResponse(suggestions=suggestions)
            )
    except Exception as e:
        logger.warning(f"Redis cache error: {e}")
    
    # Get top queries from search logs
    result = await db.execute(
        select(SearchLog.query, func.count(SearchLog.id).label('frequency'))
        .where(SearchLog.query.ilike(f"{q}%"))
        .group_by(SearchLog.query)
        .order_by(desc('frequency'))
        .limit(limit)
    )
    top_queries = result.all()
    
    # Get popular locations matching the query
    location_result = await db.execute(
        select(Property.location)
        .where(Property.location.ilike(f"%{q}%"))
        .distinct()
        .limit(limit - len(top_queries))
    )
    locations = location_result.scalars().all()
    
    # Combine suggestions
    suggestions = [query for query, _ in top_queries] + locations
    suggestions = suggestions[:limit]
    
    if not suggestions:
        # Fallback to location suggestions
        result = await db.execute(
            select(Property.location)
            .distinct()
            .order_by(Property.location)
            .limit(limit)
        )
        suggestions = result.scalars().all()
    
    # Cache for 1 hour
    try:
        await redis_client.setex(cache_key, 3600, json.dumps(suggestions))
    except Exception as e:
        logger.warning(f"Failed to cache suggestions: {e}")
    
    logger.success(f"Retrieved {len(suggestions)} suggestions")
    
    return StandardResponse(
        message="Suggestions retrieved successfully",
        data=SuggestionsResponse(suggestions=suggestions)
    )


@router.get("/similar/{property_id}", response_model=StandardResponse[SimilarPropertiesResponse], tags=["Search"])
async def get_similar_properties(
    property_id: uuid.UUID,
    limit: int = Query(6, ge=1, le=12),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns properties semantically similar to the given property.
    Uses stored embeddings for efficient similarity search.
    """
    logger.info(f"Similar properties requested for: {property_id}")
    
    # Get source property
    result = await db.execute(
        select(Property).where(Property.id == property_id)
    )
    source_property = result.scalars().first()
    
    if not source_property:
        raise HTTPException(status_code=404, detail="Property not found")
    
    # Get all approved properties (excluding source)
    result = await db.execute(
        select(Property)
        .where(and_(
            Property.status == PropertyStatus.APPROVED,
            Property.id != property_id
        ))
    )
    properties = result.scalars().all()
    
    # Calculate similarity based on amenities, bedrooms, location, price range
    similar_with_scores = []
    
    for prop in properties:
        similarity_score = 0.5  # Base score
        
        # Bedroom match
        if prop.bedrooms == source_property.bedrooms:
            similarity_score += 0.2
        elif abs(prop.bedrooms - source_property.bedrooms) <= 1:
            similarity_score += 0.1
        
        # Location proximity (simple string matching)
        if prop.location.lower() == source_property.location.lower():
            similarity_score += 0.3
        elif prop.location.split(',')[0].lower() == source_property.location.split(',')[0].lower():
            similarity_score += 0.15
        
        # Price similarity
        price_diff = abs(prop.price - source_property.price) / max(source_property.price, prop.price)
        if price_diff < 0.2:
            similarity_score += 0.2
        elif price_diff < 0.5:
            similarity_score += 0.1
        
        # Amenities overlap
        source_amenities = set(source_property.amenities or [])
        prop_amenities = set(prop.amenities or [])
        if source_amenities and prop_amenities:
            overlap = len(source_amenities & prop_amenities) / len(source_amenities | prop_amenities)
            similarity_score += overlap * 0.2
        
        similarity_score = min(1.0, similarity_score)
        similar_with_scores.append((prop, similarity_score))
    
    # Sort by similarity score
    similar_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get top results
    from schemas.search import SimilarProperty
    similar_properties = [
        SimilarProperty(
            id=prop.id,
            title=prop.title,
            price=prop.price,
            price_type=prop.price_type,
            location=prop.location,
            thumbnail=prop.thumbnail,
            similarity_score=min(1.0, score)
        )
        for prop, score in similar_with_scores[:limit]
    ]
    
    logger.success(f"Retrieved {len(similar_properties)} similar properties")
    
    return StandardResponse(
        message="Similar properties retrieved successfully",
        data=SimilarPropertiesResponse(
            source_property_id=property_id,
            similar_properties=similar_properties
        )
    )


@router.post("/feedback", response_model=StandardResponse[SearchFeedbackResponse], status_code=201, tags=["Search"])
async def log_search_feedback(
    feedback: SearchFeedbackRequest,
    current_user: Optional[User] = Depends(lambda: None),
    db: AsyncSession = Depends(get_db)
):
    """
    Record user feedback on a search result for model evaluation and improvement.
    """
    logger.info(f"Search feedback received - Type: {feedback.feedback_type}")
    
    # Verify listing exists
    result = await db.execute(
        select(Property).where(Property.id == feedback.listing_id)
    )
    if not result.scalars().first():
        raise HTTPException(status_code=404, detail="Listing not found")
    
    # Create feedback record
    feedback_record = SearchFeedback(
        user_id=current_user.id if current_user else None,
        session_id=feedback.session_id,
        query=feedback.query,
        listing_id=feedback.listing_id,
        feedback_type=feedback.feedback_type,
        explanation_helpful=feedback.explanation_helpful,
        comments=feedback.comments
    )
    
    db.add(feedback_record)
    await db.commit()
    await db.refresh(feedback_record)
    
    logger.success(f"Feedback recorded - ID: {feedback_record.id}")
    
    return StandardResponse(
        message="Feedback recorded. Thank you!",
        data=SearchFeedbackResponse(
            message="Feedback recorded successfully",
            feedback_id=feedback_record.id
        )
    )
