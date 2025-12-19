"""
Embedder module for generating text embeddings and building vector store
Handles vector database operations for RAG functionality
Uses OpenAI API directly instead of LangChain
"""
import os
import sys
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

# Ensure the root directory is in the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

try:
    from openai import OpenAI
except ImportError:
    print("Warning: OpenAI module not installed. Please run: pip install openai")

import config


class Document:
    """Simple document class to replace LangChain Document"""
    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(content_length={len(self.page_content)}, metadata={self.metadata})"


class OpenFOAMEmbedder:
    """Handles embedding generation and vector store operations for OpenFOAM cases"""

    def __init__(self):
        """Initialize the embedder with OpenAI client"""
        self.client = OpenAI(
            api_key=config.EMBEDDING_API_KEY,
            base_url=config.EMBEDDING_BASE_URL
        )
        self.vector_store = None
        self.documents = []
        self.embeddings_cache = []
        self.embedding_dim = None
        self._set_embedding_dim()
        
    def load_tutorial_descriptions(self) -> List[Document]:
        """
        Load all tutorial description files from the database directory

        Returns:
            List of Document objects
        """
        documents = []
        tutorial_dir = config.TUTORIAL_DESCRIPTIONS_DIR

        if not tutorial_dir.exists():
            print(f"Tutorial directory not found: {tutorial_dir}")
            return documents

        # Load markdown files
        for md_file in tutorial_dir.glob("*.md"):
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(Document(
                    page_content=content,
                    metadata={"source": md_file.name, "type": "markdown"}
                ))

        # Load JSON files
        for json_file in tutorial_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Convert JSON to readable text format
                content = self._json_to_text(data)
                documents.append(Document(
                    page_content=content,
                    metadata={"source": json_file.name, "type": "json", "data": data}
                ))

        print(f"Loaded {len(documents)} tutorial descriptions")
        return documents

    def _json_to_text(self, data: Dict[str, Any]) -> str:
        """Convert JSON tutorial description to readable text format"""
        lines = []
        lines.append(f"--- OpenFOAM Tutorial Case: {data.get('case_name', 'Unknown')} ---")
        lines.append(f"Solver: {data.get('solver', 'N/A')}")
        lines.append(f"Physics: {data.get('physics', 'N/A')}")
        lines.append(f"Geometry: {data.get('geometry', 'N/A')}")

        if 'boundary_conditions' in data:
            lines.append("\nBoundary Conditions:")
            for bc_name, bc_info in data['boundary_conditions'].items():
                lines.append(f"  - {bc_name}: {bc_info}")

        if 'initial_conditions' in data:
            lines.append("\nInitial Conditions:")
            for ic_name, ic_value in data['initial_conditions'].items():
                lines.append(f"  - {ic_name}: {ic_value}")

        if 'time_control' in data:
            lines.append("\nTime Control:")
            for key, value in data['time_control'].items():
                lines.append(f"  - {key}: {value}")

        lines.append("--- End of Case Description ---")
        return "\n".join(lines)

    def _split_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into chunks with overlap

        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # If not at the end, try to break at a newline or space
            if end < len(text):
                # Look for newline in the last 100 chars
                newline_pos = text.rfind('\n', start, end)
                if newline_pos > start + chunk_size // 2:
                    end = newline_pos + 1
                else:
                    # Look for space
                    space_pos = text.rfind(' ', start, end)
                    if space_pos > start + chunk_size // 2:
                        end = space_pos + 1

            chunks.append(text[start:end])
            start = end - overlap if end < len(text) else end

        return chunks

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks

        Args:
            documents: List of documents to split

        Returns:
            List of split documents
        """
        split_docs = []

        for doc in documents:
            chunks = self._split_text(doc.page_content, chunk_size=1000, overlap=200)

            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata['chunk_index'] = i
                metadata['total_chunks'] = len(chunks)
                split_docs.append(Document(page_content=chunk, metadata=metadata))

        return split_docs

    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text using OpenAI API

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        try:
            # Truncate text if too long (OpenAI has token limits)
            max_chars = 8000  # Approximately 2000 tokens
            if len(text) > max_chars:
                text = text[:max_chars]

            response = self.client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=text
            )

            return response.data[0].embedding

        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * self.embedding_dim

    def _set_embedding_dim(self):
        """Determine embedding dimension from the model"""
        try:
            response = self.client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input="test"
            )
            self.embedding_dim = len(response.data[0].embedding)
            if config.ENABLE_VERBOSE_LOGGING:
                print(f"Detected embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"Could not determine embedding dimension, defaulting to 1024. Error: {e}")
            self.embedding_dim = 1024

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts in batch

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        batch_size = 10  # Process in smaller batches to avoid rate limits

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                # Truncate texts if too long
                max_chars = 8000
                truncated_batch = [text[:max_chars] if len(text) > max_chars else text for text in batch]

                response = self.client.embeddings.create(
                    model=config.EMBEDDING_MODEL,
                    input=truncated_batch
                )

                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

                if config.ENABLE_VERBOSE_LOGGING:
                    print(f"Processed embeddings batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

            except Exception as e:
                print(f"Error in batch embedding: {str(e)}")
                # Add zero vectors for failed batch
                embeddings.extend([[0.0] * self.embedding_dim] * len(batch))

        return embeddings

    def build_vector_store(self, documents: List[Document] = None, force_rebuild: bool = False):
        """
        Build or load the vector store from tutorial descriptions

        Args:
            documents: Optional list of documents to index. If None, loads from files
            force_rebuild: If True, rebuilds the vector store even if it exists
        """
        persist_path = config.VECTOR_STORE_DIR / "vector_store.pkl"

        # Check if vector store already exists
        if not force_rebuild and persist_path.exists():
            print("Loading existing vector store...")
            try:
                with open(persist_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.embeddings_cache = data['embeddings']
                    self.vector_store = True
                print(f"Vector store loaded with {len(self.documents)} documents")
                return
            except Exception as e:
                print(f"Error loading vector store: {str(e)}")
                print("Rebuilding from scratch...")

        # Load documents if not provided
        if documents is None:
            documents = self.load_tutorial_descriptions()

        if not documents:
            print("No documents to index!")
            return

        # Split documents if they're too long
        split_docs = self._split_documents(documents)
        print(f"Building vector store with {len(split_docs)} document chunks...")

        # Generate embeddings for all documents
        texts = [doc.page_content for doc in split_docs]
        embeddings = self._get_embeddings_batch(texts)

        # Store documents and embeddings
        self.documents = split_docs
        self.embeddings_cache = embeddings
        self.vector_store = True

        # Persist to disk
        try:
            persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(persist_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'embeddings': self.embeddings_cache
                }, f)
            print(f"Vector store created and persisted to {persist_path}")
        except Exception as e:
            print(f"Warning: Could not persist vector store: {str(e)}")

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """
        Perform similarity search on the vector store

        Args:
            query: Search query string
            k: Number of results to return (default from config)

        Returns:
            List of relevant documents
        """
        if self.vector_store is None:
            print("Vector store not initialized. Building now...")
            self.build_vector_store()

        if not self.documents or not self.embeddings_cache:
            print("No documents in vector store")
            return []

        k = k or config.TOP_K_RESULTS

        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings_cache):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))

        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]

        # Get corresponding documents
        results = [self.documents[i] for i, _ in top_k]

        if config.ENABLE_VERBOSE_LOGGING:
            print(f"Found {len(results)} relevant documents for query: '{query[:50]}...'")
            for i, (idx, sim) in enumerate(top_k[:3]):
                print(f"  {i+1}. Similarity: {sim:.3f} - {self.documents[idx].metadata.get('source', 'Unknown')}")

        return results

    def format_rag_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a context string for the prompt

        Args:
            documents: List of retrieved documents

        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant context found in the knowledge base."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            context_parts.append(f"[Context {i} - Source: {source}]")
            context_parts.append(doc.page_content)
            context_parts.append("")  # Empty line between contexts

        return "\n".join(context_parts)


def build_vector_store(force_rebuild: bool = False):
    """
    Convenience function to build the vector store

    Args:
        force_rebuild: If True, rebuilds even if vector store exists
    """
    embedder = OpenFOAMEmbedder()
    embedder.build_vector_store(force_rebuild=force_rebuild)
    print("Vector store build complete!")


if __name__ == "__main__":
    # Build the vector store when run directly
    build_vector_store(force_rebuild=True)
