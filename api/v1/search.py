from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, and_, or_, desc, insert, text
from datetime import datetime, timezone
import uuid
import time
import re
from typing import Optional, List

from api.deps import get_current_user, get_current_user_optional
from db.session import get_db
from db.models.user import User
from db.models.property import Property
from db.models.search import SearchLog, SearchFeedback, RankingConfig
from core.enums import PropertyStatus
from schemas.search import (
    SemanticSearchRequest, SemanticSearchResponse, SearchResult,
    KeywordSearchResponse, SuggestionsResponse, SimilarPropertiesResponse,
    SearchFeedbackRequest, SearchFeedbackResponse, ExplanationFeature,
    NearbyProperty, NearbySearchResponse
)
from schemas.base import StandardResponse
from services.redis_service import redis_client
from services.embedding_service import batch_similarity_search
from services.geo_service import (
    haversine_km, bounding_box, distance_score,
    parse_geo_filters, resolve_location
)
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
        'water supply': ['water', 'borehole'],
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
    actual_location: str,
    distance_km: Optional[float] = None
) -> float:
    """
    Enhanced location scoring that considers:
    1. Explicit filter location (highest priority)
    2. Location mentions in semantic query text
    3. Geographic proximity via exact haversine distance (when resolved)
    4. String matching as fallback
    """
    # String matching: Explicit filter location has highest priority
    if filter_location and filter_location.lower() in actual_location.lower():
        return 0.95

    # Check if any query-mentioned locations match
    if query_mentioned_locations:
        for mentioned_loc in query_mentioned_locations:
            if mentioned_loc.lower() in actual_location.lower():
                return 0.85  # High score for query-mentioned location

    # Geographic proximity: smooth decay over exact distance from the
    # resolved search center (see services.geo_service.distance_score)
    if distance_km is not None:
        return distance_score(distance_km)

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
    current_user: Optional[User] = Depends(get_current_user_optional),
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

    # GEO: resolve a search center from explicit coordinates or the filter location
    geo_center = None
    geo_radius_km = None
    if request.filters:
        try:
            geo_center, geo_radius_km = parse_geo_filters(request.filters)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if geo_center is None and request.filters.get('location'):
            resolved = await resolve_location(db, request.filters['location'])
            if resolved:
                geo_center = (resolved.lat, resolved.lng)
                logger.debug(f"Geocoded filter location '{request.filters['location']}' via {resolved.source}")
            elif geo_radius_km is not None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not resolve location '{request.filters['location']}' for radius search. Provide lat/lng instead."
                )

    # Radius filter: indexed bounding-box prefilter in SQL, exact haversine refinement after fetch
    if geo_center and geo_radius_km:
        lat_min, lat_max, lng_min, lng_max = bounding_box(geo_center[0], geo_center[1], geo_radius_km)
        filters.append(Property.latitude.between(lat_min, lat_max))
        filters.append(Property.longitude.between(lng_min, lng_max))
        logger.debug(f"Applied radius filter: {geo_radius_km}km around ({geo_center[0]:.4f}, {geo_center[1]:.4f})")

    # Get dynamic ranking weights (adjusted based on query content)
    weights = await get_dynamic_weights(db, request.query)
    
    # Fetch all matching properties
    result = await db.execute(
        select(Property).where(and_(*filters)).limit(1000)  # Batch processing limit
    )
    properties = result.scalars().all()

    # Exact radius refinement: the bounding box over-selects at its corners
    if geo_center and geo_radius_km:
        properties = [
            p for p in properties
            if p.latitude is not None and p.longitude is not None
            and haversine_km(geo_center[0], geo_center[1], p.latitude, p.longitude) <= geo_radius_km
        ]

    if not properties:
        return StandardResponse(
            message="Search results retrieved successfully",
            data=SemanticSearchResponse(
                query=request.query,
                total_results=0,
                page=request.page,
                limit=request.limit,
                processing_time_ms=int((time.time() - query_start) * 1000),
                results=[]
            )
        )

    # Extract locations mentioned in the semantic query for enhanced scoring
    all_unique_locations = list(set(p.location for p in properties if p.location))
    query_mentioned_locations = extract_locations_from_query(request.query, all_unique_locations)

    # No explicit geo center: resolve one from the query text for distance-aware ranking
    if geo_center is None and query_mentioned_locations:
        resolved = await resolve_location(db, query_mentioned_locations[0])
        if resolved:
            geo_center = (resolved.lat, resolved.lng)
            logger.debug(f"Ranking distance center from query location '{query_mentioned_locations[0]}'")
    
    # Use BATCH similarity search for 10-50x performance improvement!
    candidates = [
        f"{prop.title}. {prop.description}" for prop in properties
    ]
    
    # Batch search returns top results efficiently
    batch_start = time.time()
    try:
        batch_results = batch_similarity_search(
            query=request.query,
            candidates=candidates,
            top_k=min(len(candidates), 100),  # Get top 100 results
            normalize=True
        )
        batch_time_ms = int((time.time() - batch_start) * 1000)
        logger.debug(f"Batch similarity search: {len(batch_results)} results in {batch_time_ms}ms")
    except Exception as exc:
        batch_time_ms = int((time.time() - batch_start) * 1000)
        logger.warning(f"Semantic embedding backend failed after {batch_time_ms}ms: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Semantic search is temporarily unavailable while the embedding service is unreachable. Please try again shortly."
        ) from exc

    # Calculate scores for each result from batch
    scored_properties = []
    for idx, _, batch_semantic_score in batch_results:
        prop = properties[idx]
        semantic_score = batch_semantic_score
        price_score = calculate_price_score(
            request.filters.get('price_max') if request.filters else None,
            prop.price
        )
        # Exact distance from the resolved search center, if we have one
        prop_distance_km = None
        if geo_center and prop.latitude is not None and prop.longitude is not None:
            prop_distance_km = round(
                haversine_km(geo_center[0], geo_center[1], prop.latitude, prop.longitude), 2
            )
        # Use enhanced location scoring with query-extracted locations and geo-coordinates
        location_score = calculate_location_score_enhanced(
            query_mentioned_locations=query_mentioned_locations,
            filter_location=request.filters.get('location') if request.filters else None,
            actual_location=prop.location,
            distance_km=prop_distance_km
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
                    label=(
                        f"About {prop_distance_km} km from your search area" if prop_distance_km is not None and location_score <= 0.9
                        else "Located in your preferred area" if location_score > 0.9
                        else ("Partially matches your location" if location_score > 0.6 else "Different location")
                    ),
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
            'distance_km': prop_distance_km,
            'final_score': final_score_boosted,  # Use boosted score
            'explanations': explanations,
            'explanation_summary': explanation_summary
        })

    # Sort by relevance before pagination
    scored_properties.sort(key=lambda item: item['final_score'], reverse=True)
    
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
            distance_km=item['distance_km'],
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
            distance_km=item['distance_km'],
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
    current_user: Optional[User] = Depends(get_current_user_optional),
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


@router.get("/nearby", response_model=StandardResponse[NearbySearchResponse], tags=["Search"])
async def nearby_search(
    lat: Optional[float] = Query(None, ge=-90, le=90),
    lng: Optional[float] = Query(None, ge=-180, le=180),
    location: Optional[str] = Query(None, min_length=2, max_length=255, description="Free-text location, geocoded when lat/lng are not provided"),
    radius_km: float = Query(5.0, gt=0, le=500),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db)
):
    """
    Radius/proximity search: approved properties within radius_km of a point,
    sorted by distance. The center comes from lat/lng, or from geocoding a
    free-text location.
    """
    query_start = time.time()

    if (lat is None) != (lng is None):
        raise HTTPException(status_code=400, detail="Both 'lat' and 'lng' must be provided together")

    resolved_from = None
    if lat is None:
        if not location:
            raise HTTPException(status_code=400, detail="Provide either 'lat'/'lng' or a 'location' to search around")
        resolved = await resolve_location(db, location)
        if not resolved:
            raise HTTPException(status_code=404, detail=f"Could not resolve location '{location}'")
        lat, lng = resolved.lat, resolved.lng
        resolved_from = location

    logger.info(f"Nearby search initiated - Center: ({lat:.4f}, {lng:.4f}), Radius: {radius_km}km")

    # Indexed bounding-box prefilter, then exact haversine refinement
    lat_min, lat_max, lng_min, lng_max = bounding_box(lat, lng, radius_km)
    result = await db.execute(
        select(Property).where(and_(
            Property.status == PropertyStatus.APPROVED,
            Property.latitude.between(lat_min, lat_max),
            Property.longitude.between(lng_min, lng_max)
        ))
    )
    properties = result.scalars().all()

    within_radius = []
    for prop in properties:
        distance = haversine_km(lat, lng, prop.latitude, prop.longitude)
        if distance <= radius_km:
            within_radius.append((prop, distance))

    within_radius.sort(key=lambda x: x[1])

    total_results = len(within_radius)
    offset = (page - 1) * limit
    paginated = within_radius[offset:offset + limit]

    results = [
        NearbyProperty(
            id=prop.id,
            title=prop.title,
            price=prop.price,
            price_type=prop.price_type,
            location=prop.location,
            bedrooms=prop.bedrooms,
            bathrooms=prop.bathrooms,
            thumbnail=prop.thumbnail,
            distance_km=round(distance, 2)
        )
        for prop, distance in paginated
    ]

    processing_time_ms = int((time.time() - query_start) * 1000)

    background_tasks.add_task(
        log_search_async,
        db,
        current_user.id if current_user else None,
        location or f"({lat:.4f}, {lng:.4f})",
        {"lat": lat, "lng": lng, "radius_km": radius_km},
        total_results,
        "nearby",
        processing_time_ms
    )

    logger.success(f"Nearby search completed - Results: {total_results}, Time: {processing_time_ms}ms")

    return StandardResponse(
        message="Nearby properties retrieved successfully",
        data=NearbySearchResponse(
            center_lat=lat,
            center_lng=lng,
            radius_km=radius_km,
            resolved_from=resolved_from,
            total_results=total_results,
            page=page,
            limit=limit,
            results=results
        )
    )


@router.get("/suggestions", response_model=StandardResponse[SuggestionsResponse], tags=["Search"])
async def get_search_suggestions(
    q: str = Query(..., min_length=2, max_length=500),
    limit: int = Query(5, ge=1, le=10),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns query autocompletion suggestions based on popular past queries.
    Filters by queries that start with the input string (case-insensitive).
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
    
    # Get popular queries that match the input from search logs
    result = await db.execute(
        select(SearchLog.query, func.count(SearchLog.id).label('frequency'))
        .where(SearchLog.query.ilike(f"{q}%"))  # Match queries that START with input
        .group_by(SearchLog.query)
        .order_by(desc('frequency'))
        .limit(limit)
    )
    top_queries = result.all()
    
    # Extract just the query strings
    suggestions = [query for query, _ in top_queries]
    
    # If we don't have enough suggestions, get more by partial matching
    if len(suggestions) < limit:
        result = await db.execute(
            select(SearchLog.query, func.count(SearchLog.id).label('frequency'))
            .where(SearchLog.query.ilike(f"%{q}%"))  # Partial match
            .group_by(SearchLog.query)
            .order_by(desc('frequency'))
            .limit(limit - len(suggestions))
        )
        additional_queries = result.all()
        suggestions.extend([query for query, _ in additional_queries if query not in suggestions])
    
    # If still no suggestions, return empty list (not fallback to locations)
    if not suggestions:
        logger.debug(f"No matching queries found for: {q}")
    
    # Cache for 1 hour
    try:
        await redis_client.setex(cache_key, 3600, json.dumps(suggestions))
    except Exception as e:
        logger.warning(f"Failed to cache suggestions: {e}")
    
    logger.success(f"Retrieved {len(suggestions)} suggestions for: {q}")
    
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
        
        # Location proximity: exact distance when both have coordinates,
        # string matching as fallback
        if (prop.latitude is not None and prop.longitude is not None and
                source_property.latitude is not None and source_property.longitude is not None):
            distance = haversine_km(
                source_property.latitude, source_property.longitude,
                prop.latitude, prop.longitude
            )
            if distance <= 2:
                similarity_score += 0.3
            elif distance <= 10:
                similarity_score += 0.15
        elif prop.location.lower() == source_property.location.lower():
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
    current_user: Optional[User] = Depends(get_current_user_optional),
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
