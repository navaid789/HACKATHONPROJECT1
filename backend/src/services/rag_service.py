import asyncio
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
from sqlalchemy.orm import Session
import logging
from datetime import datetime

from ..models.content import Content
from ..models.content_embedding import ContentEmbedding
from ..config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            https=settings.qdrant_https
        )

        # Initialize sentence transformer model for embeddings
        self.embedding_model = SentenceTransformer(settings.embedding_model)

        # Initialize text generation pipeline for responses
        self.text_generator = pipeline(
            "text-generation",
            model=settings.llm_model,
            device="cpu"  # Use CPU by default, can be changed based on availability
        )

        # Create collection in Qdrant if it doesn't exist
        self._create_collection()

    def _create_collection(self):
        """
        Create Qdrant collection for content embeddings if it doesn't exist
        """
        try:
            self.qdrant_client.get_collection("content_embeddings")
        except:
            # Collection doesn't exist, create it
            self.qdrant_client.create_collection(
                collection_name="content_embeddings",
                vectors_config=models.VectorParams(
                    size=self.embedding_model.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE
                )
            )
            logger.info("Created Qdrant collection for content embeddings")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a given text
        """
        embedding = self.embedding_model.encode([text])
        return embedding[0].tolist()  # Convert to list for JSON serialization

    async def add_content_to_vector_db(self, content_id: int, text: str, metadata: Optional[Dict] = None):
        """
        Add content to the vector database for retrieval
        """
        try:
            # Generate embedding
            embedding = self.generate_embedding(text)

            # Prepare payload
            payload = {
                "content_id": content_id,
                "text": text,
                "created_at": datetime.utcnow().isoformat()
            }

            if metadata:
                payload.update(metadata)

            # Add to Qdrant
            self.qdrant_client.upsert(
                collection_name="content_embeddings",
                points=[
                    models.PointStruct(
                        id=content_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )

            logger.info(f"Added content {content_id} to vector database")
            return True
        except Exception as e:
            logger.error(f"Error adding content to vector database: {str(e)}")
            return False

    async def search_similar_content(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar content based on query
        """
        try:
            # Generate embedding for query
            query_embedding = self.generate_embedding(query)

            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name="content_embeddings",
                query_vector=query_embedding,
                limit=top_k
            )

            # Format results
            results = []
            for result in search_results:
                results.append({
                    "content_id": result.payload.get("content_id"),
                    "text": result.payload.get("text"),
                    "score": result.score,
                    "metadata": {k: v for k, v in result.payload.items()
                               if k not in ["content_id", "text", "created_at"]}
                })

            return results
        except Exception as e:
            logger.error(f"Error searching similar content: {str(e)}")
            return []

    async def query(self, query: str, user_id: Optional[int] = None, top_k: int = 3) -> Dict[str, Any]:
        """
        Main query method that retrieves relevant content and generates response
        """
        try:
            # Search for relevant content
            relevant_content = await self.search_similar_content(query, top_k)

            # Prepare context from retrieved content
            context = ""
            content_ids = []

            for item in relevant_content:
                context += f"Content: {item['text']}\n\n"
                content_ids.append(item['content_id'])

            # If no relevant content found, return a message
            if not context.strip():
                return {
                    "response": "I couldn't find any relevant content to answer your question. Please try rephrasing your query or check if the content exists in the textbook.",
                    "sources": [],
                    "confidence": 0.0
                }

            # Prepare prompt for text generation
            prompt = f"""
            You are an AI assistant for the Physical AI & Humanoid Robotics Textbook.
            Use the following context to answer the user's question.

            Context:
            {context}

            User question: {query}

            Please provide a helpful and accurate response based on the context.
            If the context doesn't contain enough information to answer the question,
            politely explain that the information might not be available in the current content.
            """

            # Generate response using the text generation pipeline
            response = self.text_generator(
                prompt,
                max_length=500,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )

            # Extract the generated text
            generated_text = response[0]['generated_text']

            # Extract the answer part (remove the prompt)
            if "User question:" in generated_text:
                answer = generated_text.split("User question:")[1].split("\n")[1:]
                answer = "\n".join(answer).strip()
            else:
                answer = generated_text.strip()

            return {
                "response": answer,
                "sources": content_ids,
                "confidence": 0.8  # Placeholder confidence score
            }

        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return {
                "response": "I encountered an error while processing your request. Please try again.",
                "sources": [],
                "confidence": 0.0
            }

    async def generate_content_summary(self, content: str, max_length: int = 150) -> str:
        """
        Generate a summary of the given content
        """
        try:
            prompt = f"""
            Please provide a concise summary of the following content:

            {content}

            Summary:
            """

            response = self.text_generator(
                prompt,
                max_length=max_length,
                min_length=30,
                num_return_sequences=1,
                temperature=0.5,
                do_sample=True
            )

            summary = response[0]['generated_text']

            # Extract the summary part (remove the prompt)
            if "Summary:" in summary:
                summary = summary.split("Summary:")[1].strip()
            else:
                summary = summary.strip()

            return summary

        except Exception as e:
            logger.error(f"Error generating content summary: {str(e)}")
            return "Summary generation failed."

    async def evaluate_answer_relevance(self, question: str, answer: str, context: str) -> float:
        """
        Evaluate how relevant the answer is to the question given the context
        """
        try:
            prompt = f"""
            Evaluate how well the following answer addresses the question given the context.
            Rate the relevance on a scale of 0 to 1 (0 = not relevant, 1 = highly relevant).

            Question: {question}
            Context: {context}
            Answer: {answer}

            Relevance score (0-1):
            """

            response = self.text_generator(
                prompt,
                max_length=100,
                num_return_sequences=1,
                temperature=0.1,
                do_sample=False
            )

            evaluation = response[0]['generated_text']

            # Extract the score (look for numbers between 0 and 1)
            import re
            score_match = re.search(r'(\d+\.?\d*)', evaluation)
            if score_match:
                score = float(score_match.group(1))
                return min(1.0, max(0.0, score))  # Ensure score is between 0 and 1
            else:
                return 0.5  # Default score if parsing fails

        except Exception as e:
            logger.error(f"Error evaluating answer relevance: {str(e)}")
            return 0.5