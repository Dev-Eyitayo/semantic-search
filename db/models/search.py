import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Text, Float, DateTime, Enum, ForeignKey, JSON, Integer
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import Mapped, mapped_column
from db.base import Base


class SearchLog(Base):
    """Stores search query logs for analytics and model training"""
    __tablename__ = "search_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    query: Mapped[str] = mapped_column(String(500), nullable=False)
    filters: Mapped[dict] = mapped_column(JSON, nullable=True)
    results_count: Mapped[int] = mapped_column(Integer, default=0)
    search_type: Mapped[str] = mapped_column(String(50), nullable=False)  # semantic, keyword, fallback
    processing_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class SearchFeedback(Base):
    """Stores user feedback on search results for model evaluation"""
    __tablename__ = "search_feedback"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    session_id: Mapped[str] = mapped_column(String(255), nullable=True, index=True)
    query: Mapped[str] = mapped_column(String(500), nullable=False)
    listing_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("properties.id", ondelete="CASCADE"), nullable=False)
    feedback_type: Mapped[str] = mapped_column(String(50), nullable=False)  # thumbs_up, thumbs_down, irrelevant
    explanation_helpful: Mapped[bool] = mapped_column(default=None, nullable=True)
    comments: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class RankingConfig(Base):
    """Stores the hybrid ranking formula weights for search results"""
    __tablename__ = "ranking_config"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    semantic_score_weight: Mapped[float] = mapped_column(Float, default=0.55)
    price_score_weight: Mapped[float] = mapped_column(Float, default=0.20)
    location_score_weight: Mapped[float] = mapped_column(Float, default=0.15)
    recency_score_weight: Mapped[float] = mapped_column(Float, default=0.10)
    embedding_model: Mapped[str] = mapped_column(String(100), default="all-mpnet-base-v2")
    reranker_model: Mapped[str] = mapped_column(String(100), default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class AdminAuditLog(Base):
    """Stores audit logs for admin actions and ranking decisions"""
    __tablename__ = "admin_audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    admin_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action: Mapped[str] = mapped_column(String(255), nullable=False)  # audit_query, config_update, reindex, etc.
    query: Mapped[str] = mapped_column(Text, nullable=True)
    listing_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("properties.id", ondelete="SET NULL"), nullable=True)
    details: Mapped[dict] = mapped_column(JSON, nullable=True)
    bias_flags: Mapped[list] = mapped_column(JSON, default=[])
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
