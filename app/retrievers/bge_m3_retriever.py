"""
BGE-M3-based retriever for table search using dense embeddings.

Uses BAAI/bge-m3 model for encoding queries and documents into dense vectors,
then performs similarity search. Automatically downloads model if not present locally.
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from .base import TableRetriever

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


class BGEM3Retriever(TableRetriever):
    """
    BGE-M3-based retriever for table search using dense embeddings.

    Uses BAAI/bge-m3 model for encoding queries and documents into dense vectors,
    then performs similarity search. Automatically downloads model if not present locally.
    """

    def __init__(self, model_dir: str = None,
                 index_dir: str = None):
        """
        Args:
            model_dir: Directory to store/load the BGE-M3 model
            index_dir: Directory to store/load vector indices
        """
        project_root = Path(__file__).resolve().parents[2]
        self.model_dir = model_dir or str(project_root / "models")
        self.index_dir = index_dir or str(project_root / "indexes")
        self.model = None
        self.tokenizer = None
        self.embeddings = None
        self.table_docs = None
        self.loaded = False
        self.current_index_key = None
        self.model_name = "BAAI/bge-m3"

        # Check GPU availability and user preference
        self.use_gpu = os.getenv('USE_GPU', 'false').lower() == 'true'
        if self.use_gpu:
            try:
                import torch
                # Check for CUDA first, then MPS (for Apple Silicon), then CPU
                if torch.cuda.is_available():
                    self.device = torch.device('cuda')
                elif torch.backends.mps.is_available():
                    self.device = torch.device('mps')
                else:
                    self.device = torch.device('cpu')
                    print("Warning: USE_GPU=true but neither CUDA nor MPS available, falling back to CPU")
            except ImportError:
                print("Warning: PyTorch not available, falling back to CPU")
                self.device = 'cpu'
                self.use_gpu = False
        else:
            self.device = 'cpu'

        print(f"BGE-M3 will use device: {self.device}")

        # Ensure directories exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)

    def load_model(self, model_path: str = None):
        """Load the BGE-M3 model using FlagEmbedding."""
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ImportError("FlagEmbedding library is required for BGEM3Retriever. Install with: pip install FlagEmbedding")

        model_path = model_path or self.model_name

        print(f"Loading BGE-M3 model using FlagEmbedding: {model_path}")
        
        # Determine device
        device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        
        try:
            self.model = BGEM3FlagModel(model_path, use_fp16=True, devices=device)
            print(f"BGE-M3 model loaded successfully on device: {device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load BGE-M3 model: {e}")

        self.loaded = True

    def unload_model(self):
        """Unload the model to free GPU/CPU memory."""
        self.model = None
        self.embeddings = None
        self.table_docs = None
        self.loaded = False
        self.current_index_key = None
        print("BGE-M3 model unloaded")

    def _get_index_key(self, dataset_category: str, dataset_name: str, split: str,
                       representation_type: str, representation_params: Dict[str, Any]) -> str:
        """
        Generate a unique key for the index based on dataset and representation parameters.
        """
        # Create a sorted parameter string for consistency
        param_str = "_".join([f"{k}={v}" for k, v in sorted(representation_params.items())])
        if param_str:
            param_str = f"_{param_str}"

        key = f"bge_m3_{dataset_category}_{dataset_name}_{split}_{representation_type}{param_str}"
        return key

    def _get_index_path(self, index_key: str) -> str:
        """Get the file path for a given index key."""
        return os.path.join(self.index_dir, f"{index_key}.pkl")

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode a single text into a dense vector using BGE-M3."""
        # Use FlagEmbedding's encode method with maximum supported length for BGE-M3 (8192 tokens)
        result = self.model.encode([text], max_length=8192)
        embedding = result['dense_vecs'][0]
        
        # Convert to numpy array if not already
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        elif hasattr(embedding, 'tolist'):
            embedding = embedding.astype(np.float32)
        
        return embedding

    def encode_indices(self, representations: List[str], table_ids: List[Any] = None, **kwargs) -> str:
        """
        Build dense vector index from text representations using BGE-M3.

        Args:
            representations: List of text representations of tables
            table_ids: List of table IDs corresponding to representations (optional)
            **kwargs: Additional parameters (should include dataset info for key generation)

        Returns:
            Index key for the created index
        """
        if not self.loaded:
            raise RuntimeError("Retriever not loaded")

        # Extract parameters for index key
        dataset_category = kwargs.get('dataset_category', 'unknown')
        dataset_name = kwargs.get('dataset_name', 'unknown')
        split = kwargs.get('split', 'unknown')
        representation_type = kwargs.get('representation_type', 'unknown')
        representation_params = kwargs.get('representation_params', {})

        # Generate index key
        index_key = self._get_index_key(dataset_category, dataset_name, split,
                                       representation_type, representation_params)

        # Check if index already exists
        index_path = self._get_index_path(index_key)
        if os.path.exists(index_path):
            print(f"BGE-M3 index already exists at {index_path}, loading instead of re-encoding...")
            self._load_index(index_key)
            return index_key

        print(f"Building BGE-M3 embeddings for {len(representations)} documents...")

        # Use table_ids if provided, otherwise use indices
        if table_ids is None:
            table_ids = list(range(len(representations)))

        # Encode all documents
        embeddings_list = []
        for i, rep in enumerate(representations):
            if i % 100 == 0:
                print(f"Encoding document {i+1}/{len(representations)}...")
            
            # Print first representation content to verify it's from representations file
            if i == 0:
                print(f"First representation content (first 500 chars):\n{rep[:500]}")
                print("--- End of first representation ---\n")
            
            embedding = self._encode_text(rep)
            embeddings_list.append(embedding)

        # Convert to numpy array
        self.embeddings = np.array(embeddings_list)
        self.table_docs = representations
        self.table_ids = table_ids
        self.current_index_key = index_key

        # Save index to disk
        index_data = {
            'embeddings': self.embeddings,
            'table_docs': self.table_docs,
            'table_ids': self.table_ids,
            'index_key': index_key,
            'model_name': self.model_name
        }

        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)

        print(f"BGE-M3 index saved to {index_path}")
        return index_key

    def retrieve(self, query: str, index_key: str, top_k: int = 5, **kwargs) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve relevant documents using BGE-M3 embeddings and cosine similarity.

        Args:
            query: The search query
            index_key: Key of the index to use
            top_k: Number of top results to return
            **kwargs: Additional parameters

        Returns:
            List of (table_dict, score) tuples
        """
        if not self.loaded:
            raise RuntimeError("Retriever not loaded")

        # Load index if not already loaded or different index requested
        if self.current_index_key != index_key or self.embeddings is None:
            self._load_index(index_key)

        # Encode query
        query_embedding = self._encode_text(query)

        # Calculate cosine similarities
        # Normalize query embedding (already normalized in _encode_text)
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return results as (table_dict, score) tuples
        results = []
        for idx in top_indices:
            table_dict = {
                'id': self.table_ids[idx],
                'representation': self.table_docs[idx],
                'score': float(similarities[idx])
            }
            results.append((table_dict, float(similarities[idx])))

        print(f"BGE-M3 retrieved {len(results)} results for query: {query[:50]}...")
        return results

    def _load_index(self, index_key: str):
        """Load index from disk."""
        index_path = self._get_index_path(index_key)

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")

        print(f"Loading BGE-M3 index from {index_path}...")
        with open(index_path, 'rb') as f:
            index_data = pickle.load(f)

        self.embeddings = index_data['embeddings']
        self.table_docs = index_data['table_docs']
        self.table_ids = index_data.get('table_ids', list(range(len(self.table_docs))))  # Backward compatibility
        self.current_index_key = index_key

        print("BGE-M3 index loaded successfully")

    def list_available_indices(self) -> List[str]:
        """List all available BGE-M3 index keys."""
        indices = []
        for filename in os.listdir(self.index_dir):
            if filename.startswith('bge_m3_') and filename.endswith('.pkl'):
                indices.append(filename[:-4])  # Remove .pkl extension
        return indices

    def delete_index(self, index_key: str):
        """Delete a specific BGE-M3 index file."""
        index_path = self._get_index_path(index_key)
        if os.path.exists(index_path):
            os.remove(index_path)
            print(f"Deleted BGE-M3 index: {index_key}")
            # Clear current index if it was the deleted one
            if self.current_index_key == index_key:
                self.unload_model()
        else:
            print(f"BGE-M3 index not found: {index_key}")