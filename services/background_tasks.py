"""
Background tasks for asynchronous embedding generation and processing.
Integrates with FastAPI BackgroundTasks and Celery for scalable async operations.
"""

from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from loguru import logger
from datetime import datetime, timezone
import uuid

from services.embedding_service import generate_embeddings_batch
from core.celery_app import celery_app
from db.models.property import Property


# FastAPI Background Tasks (for quick async operations)

async def generate_and_store_embedding(
    property_id: uuid.UUID,
    title: str,
    description: str,
    db: AsyncSession
) -> None:
    """
    Generate embedding for a property and store it asynchronously.
    Suitable for FastAPI BackgroundTasks.
    
    Args:
        property_id: Property ID
        title: Property title
        description: Property description
        db: Database session
    """
    try:
        logger.info(f"Generating embedding for property {property_id}")
        
        # Combine text
        text = f"{title}. {description}"
        
        # Generate embedding
        embedding, processing_time_ms = generate_embeddings_batch(
            text,
            normalize=True,
            batch_size=1
        )
        
        # Get first embedding from batch result
        embedding = embedding[0] if embedding else None
        
        if not embedding:
            logger.warning(f"Embedding generation returned None for property {property_id}")
            return
        
        # Store in database
        result = await db.execute(
            select(Property).where(Property.id == property_id)
        )
        prop = result.scalars().first()
        
        if prop:
            prop.embedding = embedding
            prop.updated_at = datetime.now(timezone.utc)
            await db.commit()
            logger.success(
                f"Embedding stored for property {property_id} "
                f"(dims: {len(embedding)}, time: {processing_time_ms}ms)"
            )
        else:
            logger.warning(f"Property {property_id} not found for embedding storage")
    
    except Exception as e:
        logger.error(f"Background embedding generation failed for {property_id}: {e}")
        await db.rollback()


async def batch_generate_embeddings(
    property_ids: List[uuid.UUID],
    texts: List[str],
    db: AsyncSession
) -> None:
    """
    Generate embeddings for multiple properties in batch.
    More efficient than individual generation tasks.
    
    Args:
        property_ids: List of property IDs
        texts: List of property texts (title + description)
        db: Database session
    """
    if not property_ids or not texts:
        logger.warning("Empty property IDs or texts list")
        return
    
    if len(property_ids) != len(texts):
        logger.error("Mismatched property_ids and texts lengths")
        return
    
    try:
        logger.info(f"Batch generating embeddings for {len(texts)} properties")
        
        # Generate all embeddings at once
        embeddings, processing_time_ms = generate_embeddings_batch(
            texts,
            normalize=True,
            batch_size=32
        )
        
        # Store embeddings in database
        results = await db.execute(
            select(Property).where(Property.id.in_(property_ids))
        )
        properties = results.scalars().all()
        
        # Create mapping for quick lookup
        prop_map = {prop.id: prop for prop in properties}
        
        # Update embeddings
        stored_count = 0
        for prop_id, embedding in zip(property_ids, embeddings):
            if prop_id in prop_map:
                prop = prop_map[prop_id]
                prop.embedding = embedding
                prop.updated_at = datetime.now(timezone.utc)
                stored_count += 1
        
        await db.commit()
        logger.success(
            f"Batch stored {stored_count} embeddings "
            f"(total time: {processing_time_ms}ms)"
        )
    
    except Exception as e:
        logger.error(f"Batch embedding generation failed: {e}")
        await db.rollback()


# Celery Tasks (for long-running, distributed operations)

@celery_app.task(bind=True, max_retries=3)
def generate_embedding_task(
    self,
    property_id: str,
    title: str,
    description: str
) -> dict:
    """
    Celery task for generating and storing property embedding.
    Supports retries and distributed execution.
    
    Args:
        property_id: Property UUID as string
        title: Property title
        description: Property description
        
    Returns:
        Dictionary with task result
    """
    try:
        from db.session import SessionLocal
        
        logger.info(f"Celery task: Generating embedding for property {property_id}")
        
        # Combine text
        text = f"{title}. {description}"
        
        # Generate embedding (blocking call - Celery handles async)
        embedding, processing_time_ms = generate_embeddings_batch(
            text,
            normalize=True,
            batch_size=1
        )
        
        embedding = embedding[0] if embedding else None
        
        if not embedding:
            raise ValueError("Embedding generation failed")
        
        # Store in database using sync session
        session = SessionLocal()
        try:
            prop = session.query(Property).filter(
                Property.id == uuid.UUID(property_id)
            ).first()
            
            if prop:
                prop.embedding = embedding
                prop.updated_at = datetime.now(timezone.utc)
                session.commit()
                logger.success(f"Embedding stored via Celery: {property_id}")
                return {
                    "success": True,
                    "property_id": property_id,
                    "dims": len(embedding),
                    "time_ms": processing_time_ms
                }
            else:
                raise ValueError(f"Property {property_id} not found")
        
        finally:
            session.close()
    
    except Exception as exc:
        logger.error(f"Celery task failed: {exc}")
        
        # Retry with exponential backoff
        try:
            raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
        except self.MaxRetriesExceededError:
            logger.error(f"Max retries exceeded for property {property_id}")
            return {
                "success": False,
                "property_id": property_id,
                "error": str(exc)
            }


@celery_app.task(bind=True)
def batch_generate_embeddings_task(
    self,
    property_ids: List[str],
    texts: List[str]
) -> dict:
    """
    Celery task for batch generating embeddings.
    More efficient for bulk operations.
    
    Args:
        property_ids: List of property UUIDs as strings
        texts: List of property texts
        
    Returns:
        Dictionary with task result
    """
    try:
        from db.session import SessionLocal
        
        logger.info(f"Celery task: Batch generating {len(texts)} embeddings")
        
        if len(property_ids) != len(texts):
            raise ValueError("Mismatched property_ids and texts lengths")
        
        # Generate all embeddings
        embeddings, processing_time_ms = generate_embeddings_batch(
            texts,
            normalize=True,
            batch_size=32
        )
        
        # Store in database
        session = SessionLocal()
        try:
            stored_count = 0
            for prop_id, embedding in zip(property_ids, embeddings):
                prop = session.query(Property).filter(
                    Property.id == uuid.UUID(prop_id)
                ).first()
                
                if prop:
                    prop.embedding = embedding
                    prop.updated_at = datetime.now(timezone.utc)
                    stored_count += 1
            
            session.commit()
            
            logger.success(f"Batch stored {stored_count} embeddings via Celery")
            return {
                "success": True,
                "stored_count": stored_count,
                "total_time_ms": processing_time_ms
            }
        
        finally:
            session.close()
    
    except Exception as exc:
        logger.error(f"Batch Celery task failed: {exc}")
        return {
            "success": False,
            "error": str(exc)
        }


@celery_app.task
def reindex_all_properties_task() -> dict:
    """
    Celery task to reindex all approved properties with embeddings.
    Useful for periodic maintenance or after model updates.
    
    Returns:
        Dictionary with reindexing result
    """
    try:
        from db.session import SessionLocal
        from core.enums import PropertyStatus
        
        logger.info("Celery task: Reindexing all properties")
        
        session = SessionLocal()
        try:
            # Get all approved properties
            properties = session.query(Property).filter(
                Property.status == PropertyStatus.APPROVED
            ).all()
            
            logger.info(f"Found {len(properties)} approved properties to reindex")
            
            # Collect texts
            text_list = [
                f"{p.title}. {p.description}" for p in properties
            ]
            
            # Generate embeddings in batches
            embeddings, processing_time_ms = generate_embeddings_batch(
                text_list,
                normalize=True,
                batch_size=32
            )
            
            # Store embeddings
            updated_count = 0
            for prop, embedding in zip(properties, embeddings):
                prop.embedding = embedding
                prop.updated_at = datetime.now(timezone.utc)
                updated_count += 1
            
            session.commit()
            
            logger.success(
                f"Reindexed {updated_count} properties "
                f"(time: {processing_time_ms}ms)"
            )
            
            return {
                "success": True,
                "reindexed_count": updated_count,
                "total_time_ms": processing_time_ms
            }
        
        finally:
            session.close()
    
    except Exception as exc:
        logger.error(f"Reindexing task failed: {exc}")
        return {
            "success": False,
            "error": str(exc)
        }


# Task Utility Functions

def submit_embedding_task(
    property_id: uuid.UUID,
    title: str,
    description: str,
    use_celery: bool = False
) -> Optional[dict]:
    """
    Submit an embedding generation task.
    
    Args:
        property_id: Property ID
        title: Property title
        description: Property description
        use_celery: If True, use Celery; otherwise return task info
        
    Returns:
        Task info dict with task_id if Celery is used, None otherwise
    """
    if use_celery:
        task = generate_embedding_task.delay(
            str(property_id),
            title,
            description
        )
        logger.info(f"Submitted Celery task {task.id} for property {property_id}")
        return {
            "task_id": task.id,
            "property_id": str(property_id),
            "status": "submitted"
        }
    else:
        logger.info(f"Embedding task queued for property {property_id}")
        return None


def submit_batch_embedding_task(
    property_ids: List[uuid.UUID],
    texts: List[str],
    use_celery: bool = False
) -> Optional[dict]:
    """
    Submit a batch embedding generation task.
    
    Args:
        property_ids: List of property IDs
        texts: List of property texts
        use_celery: If True, use Celery
        
    Returns:
        Task info dict if Celery is used, None otherwise
    """
    if use_celery and len(property_ids) > 10:  # Use Celery for large batches
        task = batch_generate_embeddings_task.delay(
            [str(pid) for pid in property_ids],
            texts
        )
        logger.info(
            f"Submitted batch Celery task {task.id} for {len(property_ids)} properties"
        )
        return {
            "task_id": task.id,
            "property_count": len(property_ids),
            "status": "submitted"
        }
    else:
        logger.info(f"Batch embedding task queued for {len(property_ids)} properties")
        return None
