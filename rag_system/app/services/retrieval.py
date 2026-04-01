import os
import faiss
import pickle
import numpy as np
import json
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

from app.core.config import settings
from app.models.domain_models import DocumentChunk
from langchain_core.embeddings import Embeddings
import requests

class OpenRouterEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.url = "https://openrouter.ai/api/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        data = {"model": self.model, "input": texts}
        response = requests.post(self.url, headers=self.headers, json=data)
        response.raise_for_status()
        resp_json = response.json()
        return [item["embedding"] for item in sorted(resp_json["data"], key=lambda x: x.get("index", 0))]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

class HybridRetriever:
    def __init__(self):
        self.embeddings_model = OpenRouterEmbeddings(
            api_key=settings.OPENROUTER_API_KEY,
            model="nvidia/llama-nemotron-embed-vl-1b-v2:free"
        )
        # Dynamically detect embeddings dimension to support different models seamlessly
        self.dimension = len(self.embeddings_model.embed_query("test"))

        self.faiss_index = None
        self.bm25_corpus: List[str] = []
        self.bm25_index = None
        self.metadata_store: Dict[int, DocumentChunk] = {} # Internal ID -> Chunk
        self.next_id = 0

        self._load_indices()

    def _load_indices(self):
        # Create directories if needed
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        
        # Load FAISS
        if os.path.exists(settings.FAISS_INDEX_PATH):
            self.faiss_index = faiss.read_index(settings.FAISS_INDEX_PATH)
        else:
            self.faiss_index = faiss.IndexFlatL2(self.dimension)

        # Load BM25
        if os.path.exists(settings.BM25_INDEX_PATH):
            with open(settings.BM25_INDEX_PATH, 'rb') as f:
                self.bm25_index = pickle.load(f)
        
        # Load Metadata Store
        if os.path.exists(settings.METADATA_STORE_PATH):
            with open(settings.METADATA_STORE_PATH, 'r') as f:
                data = json.load(f)
                self.metadata_store = {
                    int(k): DocumentChunk.model_validate(v) for k, v in data.get("store", {}).items()
                }
                self.next_id = data.get("next_id", 0)
                
                # Reconstruct BM25 corpus from metadata to avoid saving corpus text twice
                # We sort keys to maintain order
                sorted_ids = sorted(self.metadata_store.keys())
                self.bm25_corpus = [self.metadata_store[i].text for i in sorted_ids]
                if not self.bm25_index and self.bm25_corpus:
                    tokenized_corpus = [doc.split() for doc in self.bm25_corpus]
                    self.bm25_index = BM25Okapi(tokenized_corpus)

    def _save_indices(self):
        faiss.write_index(self.faiss_index, settings.FAISS_INDEX_PATH)
        if self.bm25_index:
            with open(settings.BM25_INDEX_PATH, 'wb') as f:
                pickle.dump(self.bm25_index, f)
        
        with open(settings.METADATA_STORE_PATH, 'w') as f:
            dump_data = {
                "next_id": self.next_id,
                "store": {str(k): v.model_dump() for k, v in self.metadata_store.items()}
            }
            json.dump(dump_data, f)
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        
        # Generate Embeddings
        embeddings = self.embeddings_model.embed_documents(texts)
        embeddings_np = np.array(embeddings, dtype='float32')

        # Add to FAISS
        self.faiss_index.add(embeddings_np)

        # Update metadata and BM25 corpus
        start_id = self.next_id
        for i, chunk in enumerate(chunks):
            self.metadata_store[start_id + i] = chunk
            self.bm25_corpus.append(chunk.text)

        self.next_id += len(chunks)

        # Rebuild BM25 index (BM25 doesn't support incremental updates easily)
        tokenized_corpus = [doc.split() for doc in self.bm25_corpus]
        self.bm25_index = BM25Okapi(tokenized_corpus)

        self._save_indices()

    def get_all_documents(self) -> List[Dict[str, Any]]:
        unique_docs = set()
        for chunk in self.metadata_store.values():
            unique_docs.add(chunk.metadata.source)
        return [{"filename": doc} for doc in unique_docs]

    def delete_document(self, filename: str) -> bool:
        keys_to_delete = [
            k for k, v in self.metadata_store.items() 
            if v.metadata.source == filename
        ]
        
        if not keys_to_delete:
            return False

        for k in keys_to_delete:
            del self.metadata_store[k]

        sorted_ids = sorted(self.metadata_store.keys())
        self.bm25_corpus = [self.metadata_store[i].text for i in sorted_ids]
        
        if self.bm25_corpus:
            tokenized_corpus = [doc.split() for doc in self.bm25_corpus]
            self.bm25_index = BM25Okapi(tokenized_corpus)
        else:
            self.bm25_index = None

        self._save_indices()

        try:
            file_path = os.path.join(settings.DATA_DIR, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Failed to delete physical file {filename}: {e}")

        return True

    def search(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """
        Performs a hybrid search combining FAISS and BM25.
        Returns deduplicated top matching chunks.
        """
        if self.faiss_index.ntotal == 0 or not self.bm25_index:
            return []

        # 1. FAISS Search
        query_embedding = self.embeddings_model.embed_query(query)
        distances, indices = self.faiss_index.search(
            np.array([query_embedding], dtype='float32'), 
            k=settings.VECTOR_TOP_K * 2  # Retrieve more for deduplication
        )
        faiss_results = [self.metadata_store[int(idx)] for idx in indices[0] if idx in self.metadata_store]

        # 2. BM25 Search
        tokenized_query = query.split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        # Get top-k indices
        bm25_top_indices = np.argsort(bm25_scores)[-(settings.KEYWORD_TOP_K * 2):][::-1]
        bm25_results = [self.metadata_store[int(idx)] for idx in bm25_top_indices if int(idx) in self.metadata_store and bm25_scores[idx] > 0]

        # 3. Combine and Deduplicate
        seen_ids = set()
        seen_texts = set()
        final_results = []
        
        # RRF (Reciprocal Rank Fusion) or simple interleaving. We'll use simple interleaving here for speed.
        # Interleave FAISS and BM25 results, favoring vector first.
        max_len = max(len(faiss_results), len(bm25_results))
        for i in range(max_len):
            if i < len(faiss_results):
                chunk = faiss_results[i]
                if chunk.chunk_id not in seen_ids and chunk.text not in seen_texts:
                    seen_ids.add(chunk.chunk_id)
                    seen_texts.add(chunk.text)
                    final_results.append(chunk)
            
            if i < len(bm25_results):
                chunk = bm25_results[i]
                if chunk.chunk_id not in seen_ids and chunk.text not in seen_texts:
                    seen_ids.add(chunk.chunk_id)
                    seen_texts.add(chunk.text)
                    final_results.append(chunk)

            if len(final_results) >= top_k:
                break
                
        return final_results[:min(top_k, settings.FINAL_TOP_K)]

# Create a singleton instance
hybrid_retriever = HybridRetriever()

def get_hybrid_retriever():
    return hybrid_retriever
