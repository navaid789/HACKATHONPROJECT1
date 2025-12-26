from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class ContentEmbedding(Base):
    __tablename__ = "content_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    content_id = Column(Integer, ForeignKey("content.id"), nullable=False)
    embedding_text = Column(Text, nullable=False)  # The text that was embedded
    embedding_vector = Column(String, nullable=False)  # Vector as string (to be stored in Qdrant)
    embedding_model = Column(String(100), nullable=False)  # Model used for embedding
    embedding_metadata = Column(Text, nullable=True)  # Additional metadata about the embedding
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<ContentEmbedding(id={self.id}, content_id={self.content_id}, embedding_model={self.embedding_model})>"