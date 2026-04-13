from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from uuid import UUID
from datetime import datetime
from enum import Enum as PyEnum


class ExplanationFeature(BaseModel):
    """Individual feature explanation from RankSHAP"""
    feature: str = Field(..., description="Feature name")
    label: str = Field(..., description="Human-readable label")
    weight: float = Field(..., ge=0.0, le=1.0, description="Feature weight/contribution")
    direction: str = Field(..., pattern="^(positive|negative|neutral)$")
    shapley_value: Optional[float] = Field(None, description="Shapley value for feature attribution")


class SearchResult(BaseModel):
    """Individual property in semantic search results with explanations"""
    id: UUID
    title: str
    price: float
    price_type: str
    location: str
    bedrooms: int
    bathrooms: int
    thumbnail: Optional[str] = None
    ranking_score: float = Field(..., ge=0.0, le=1.0)
    semantic_score: float = Field(..., ge=0.0, le=1.0)
    price_score: float = Field(..., ge=0.0, le=1.0)
    location_score: float = Field(..., ge=0.0, le=1.0)
    recency_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    explanations: Optional[List[ExplanationFeature]] = None
    explanation_summary: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class SemanticSearchRequest(BaseModel):
    """Semantic search request with filters and explanation options"""
    query: str = Field(..., min_length=3, max_length=500)
    filters: Optional[dict] = Field(None, description="Filter criteria (price_min, price_max, location, bedrooms, etc.)")
    page: int = Field(1, ge=1)
    limit: int = Field(10, ge=1, le=50)
    explain: bool = Field(True, description="Include RankSHAP explanations in results")


class SemanticSearchResponse(BaseModel):
    """Semantic search response with pagination and processing info"""
    query: str
    total_results: int
    page: int
    limit: int
    processing_time_ms: int
    results: List[SearchResult]


class KeywordSearchResponse(BaseModel):
    """Keyword/full-text search fallback response"""
    query: str
    results: List[SearchResult]
    total: int
    page: int
    search_type: str = "keyword_fallback"


class SuggestionItem(BaseModel):
    """Single autocomplete suggestion"""
    text: str
    frequency: Optional[int] = None


class SuggestionsResponse(BaseModel):
    """Autocomplete suggestions response"""
    suggestions: List[str]


class SimilarProperty(BaseModel):
    """Similar property in results"""
    id: UUID
    title: str
    price: float
    price_type: str
    location: str
    thumbnail: Optional[str] = None
    similarity_score: float = Field(..., ge=0.0, le=1.0)


class SimilarPropertiesResponse(BaseModel):
    """Similar properties response"""
    source_property_id: UUID
    similar_properties: List[SimilarProperty]


class SearchFeedbackRequest(BaseModel):
    """User feedback on search results"""
    query: str = Field(..., min_length=3, max_length=500)
    listing_id: UUID
    feedback_type: str = Field(..., pattern="^(thumbs_up|thumbs_down|irrelevant)$")
    explanation_helpful: Optional[bool] = None
    comments: Optional[str] = Field(None, max_length=500)
    session_id: Optional[str] = None


class SearchFeedbackResponse(BaseModel):
    """Confirmation of feedback recording"""
    message: str
    feedback_id: UUID


# AI/Ranking Service Schemas

class FeatureAttributionDetail(BaseModel):
    """Detailed feature attribution from SHAP"""
    name: str
    raw_value: float
    shapley_value: float
    human_label: str
    direction: str = Field(..., pattern="^(positive|negative|neutral)$")


class DetailedExplanationResponse(BaseModel):
    """Full RankSHAP explanation breakdown"""
    listing_id: UUID
    query: str
    final_score: float = Field(..., ge=0.0, le=1.0)
    features: List[FeatureAttributionDetail]
    explanation_text: str
    computation_method: str = Field(..., pattern="^(rankshap|rule_based)$")
    processing_time_ms: Optional[int] = None


class EmbedRequest(BaseModel):
    """Request to generate embedding"""
    text: str = Field(..., min_length=1, max_length=5000)
    normalize: bool = False


class EmbedResponse(BaseModel):
    """Embedding generation response"""
    embedding: List[float]
    dimensions: int
    model: str
    processing_time_ms: int


class ReindexResponse(BaseModel):
    """Property reindexing response"""
    property_id: UUID
    message: str
    embedding_updated_at: datetime


class RankingWeights(BaseModel):
    """Ranking formula weights"""
    semantic_score: float = Field(..., ge=0.0, le=1.0)
    price_score: float = Field(..., ge=0.0, le=1.0)
    location_score: float = Field(..., ge=0.0, le=1.0)
    recency_score: float = Field(..., ge=0.0, le=1.0)

    def validate_weights(self):
        total = self.semantic_score + self.price_score + self.location_score + self.recency_score
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


class RankingConfigResponse(BaseModel):
    """Ranking configuration response"""
    weights: RankingWeights
    model: str
    reranker: str
    updated_at: datetime


class UpdateRankingConfigRequest(BaseModel):
    """Request to update ranking configuration"""
    weights: Optional[RankingWeights] = None


class AuditExplanationResponse(BaseModel):
    """Admin audit response for ranking decision"""
    query: str
    listing_id: UUID
    shap_values: dict
    ranking_position: int
    bias_flags: List[str]
    protected_proxy_features_detected: bool
    audit_timestamp: datetime
