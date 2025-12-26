import asyncio
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime
import logging

from ..models.content import Content
from ..models.content_embedding import ContentEmbedding
from .rag_service import RAGService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentEmbeddingPipeline:
    def __init__(self, db_session: Session, rag_service: RAGService):
        self.db = db_session
        self.rag_service = rag_service

    async def generate_embeddings_for_content(self, content_id: Optional[int] = None) -> bool:
        """
        Generate embeddings for all content or specific content item
        """
        try:
            if content_id:
                # Generate embeddings for specific content
                content = self.db.query(Content).filter(Content.id == content_id).first()
                if not content:
                    logger.error(f"Content with ID {content_id} not found")
                    return False

                success = await self._generate_embedding_for_single_content(content)
                return success
            else:
                # Generate embeddings for all content
                contents = self.db.query(Content).filter(Content.is_published == True).all()
                success_count = 0
                total_count = len(contents)

                for content in contents:
                    if await self._generate_embedding_for_single_content(content):
                        success_count += 1

                logger.info(f"Generated embeddings for {success_count}/{total_count} content items")
                return success_count == total_count

        except Exception as e:
            logger.error(f"Error in content embedding pipeline: {str(e)}")
            return False

    async def _generate_embedding_for_single_content(self, content: Content) -> bool:
        """
        Generate embedding for a single content item
        """
        try:
            # Prepare the text for embedding
            # Combine title and content for better context
            text_for_embedding = f"{content.title}: {content.content}" if content.content else content.title

            # Generate embedding using RAG service
            embedding_vector = self.rag_service.generate_embedding(text_for_embedding)

            # Check if embedding already exists
            existing_embedding = self.db.query(ContentEmbedding).filter(
                ContentEmbedding.content_id == content.id
            ).first()

            if existing_embedding:
                # Update existing embedding
                existing_embedding.embedding_text = text_for_embedding
                existing_embedding.embedding_vector = str(embedding_vector)
                existing_embedding.embedding_model = self.rag_service.embedding_model
                existing_embedding.updated_at = datetime.utcnow()
            else:
                # Create new embedding record
                embedding_record = ContentEmbedding(
                    content_id=content.id,
                    embedding_text=text_for_embedding,
                    embedding_vector=str(embedding_vector),
                    embedding_model=self.rag_service.embedding_model
                )
                self.db.add(embedding_record)

            # Commit to database
            self.db.commit()

            # Add to vector database (Qdrant)
            metadata = {
                "title": content.title,
                "module": content.module,
                "week": content.week,
                "content_type": content.content_type
            }

            await self.rag_service.add_content_to_vector_db(
                content_id=content.id,
                text=text_for_embedding,
                metadata=metadata
            )

            logger.info(f"Generated embedding for content ID {content.id}")
            return True

        except Exception as e:
            logger.error(f"Error generating embedding for content {content.id}: {str(e)}")
            # Rollback the transaction in case of error
            self.db.rollback()
            return False

    async def batch_process_content(self, batch_size: int = 10) -> bool:
        """
        Process content in batches to avoid memory issues
        """
        try:
            # Get total count of published content
            total_count = self.db.query(Content).filter(Content.is_published == True).count()

            offset = 0
            processed_count = 0

            while offset < total_count:
                # Get a batch of content
                batch = self.db.query(Content).filter(
                    Content.is_published == True
                ).offset(offset).limit(batch_size).all()

                if not batch:
                    break

                # Process each content in the batch
                for content in batch:
                    if await self._generate_embedding_for_single_content(content):
                        processed_count += 1

                offset += batch_size
                logger.info(f"Processed batch: {offset}/{total_count} total items")

            logger.info(f"Batch processing completed: {processed_count}/{total_count} items processed successfully")
            return True

        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return False

    async def update_embeddings_for_changed_content(self) -> Dict[str, Any]:
        """
        Update embeddings for content that has been modified since last embedding
        """
        try:
            # Find content that has been updated since the last embedding
            result = self.db.execute(text("""
                SELECT c.id, c.updated_at, ce.updated_at as embedding_updated_at
                FROM content c
                LEFT JOIN content_embeddings ce ON c.id = ce.content_id
                WHERE c.is_published = true
                AND (ce.updated_at IS NULL OR c.updated_at > ce.updated_at)
            """))

            content_to_update = result.fetchall()
            updated_count = 0

            for row in content_to_update:
                content_id = row[0]
                content = self.db.query(Content).filter(Content.id == content_id).first()

                if content and await self._generate_embedding_for_single_content(content):
                    updated_count += 1

            return {
                "success": True,
                "updated_count": updated_count,
                "total_outdated": len(content_to_update)
            }

        except Exception as e:
            logger.error(f"Error updating embeddings for changed content: {str(e)}")
            return {
                "success": False,
                "updated_count": 0,
                "error": str(e)
            }

    async def delete_embeddings_for_content(self, content_id: int) -> bool:
        """
        Delete embeddings for a specific content item
        """
        try:
            # Remove from database
            self.db.query(ContentEmbedding).filter(
                ContentEmbedding.content_id == content_id
            ).delete()

            # Commit to database
            self.db.commit()

            # Remove from vector database (Qdrant)
            # Note: In Qdrant, we'd typically mark as deleted rather than physically removing
            # For now, we'll just log this operation
            logger.info(f"Marked embeddings for deletion for content ID {content_id}")

            return True

        except Exception as e:
            logger.error(f"Error deleting embeddings for content {content_id}: {str(e)}")
            self.db.rollback()
            return False

    async def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about embeddings
        """
        try:
            total_content = self.db.query(Content).count()
            content_with_embeddings = self.db.query(ContentEmbedding).count()
            published_content = self.db.query(Content).filter(Content.is_published == True).count()
            published_with_embeddings = self.db.query(Content).join(
                ContentEmbedding, Content.id == ContentEmbedding.content_id
            ).filter(Content.is_published == True).count()

            return {
                "total_content": total_content,
                "content_with_embeddings": content_with_embeddings,
                "published_content": published_content,
                "published_with_embeddings": published_with_embeddings,
                "embedding_coverage": (
                    published_content > 0 and
                    round((published_with_embeddings / published_content) * 100, 2) or 0
                )
            }

        except Exception as e:
            logger.error(f"Error getting embedding stats: {str(e)}")
            return {"error": str(e)}