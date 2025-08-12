import h5py
import faiss
import threading
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, List
from .cbir import logger
from .tasks import database, store_in_redis


class EmbeddingsStore:
    def __init__(self, database):
        self.database = database
        self._store = self.database /  "embeddings_info.hdf5"
        # Paths initialized via ImageProcessor
        self.img_data = self.database / "img_repo"  / "img_data"
        self.extracted_faces_embeddings_path = self.database / "img_repo" / "extracted_faces_embeddings"
        self._lock = threading.Lock()   

    def _check_store(self):
        if not self._store.exists():
            raise ValueError("No external embedding store available")

    def _read(self) -> Tuple[np.ndarray, List[str]]:
        self._check_store()
        with h5py.File(self._store, 'r') as file:
            features = file['embeddings'][:]
            img_names = [img_name.decode('utf-8') for img_name in file['img_names'][:]]
        return features, img_names

    def _write(self, features: np.ndarray, img_names: List[str]):
        with self._lock:
            with h5py.File(self._store, 'w') as file:
                file.create_dataset('embeddings', data=features)
                dt = h5py.string_dtype(encoding='utf-8')
                file.create_dataset('img_names', data=img_names, dtype=dt)

    def _append(self, query_feature, query_img_path: str):
        features, img_names = self._read()
        query_img_name = str(Path(query_img_path).name)
        # Skip if already in store
        if query_img_name in img_names:
            logger.info(f"{query_img_name} already exists in embedding store. Skipping append.")
            return
        features = np.vstack([features, query_feature])    
        img_names.append(query_img_name)
        self._write(features, img_names)
        logger.info(f"SUCCESS: {query_img_name} successfully added to embedding store.")

    def load_allfaces_embeddings(self, external=None) -> Tuple[np.ndarray, List[str]]: 
        #load external embeddings
        if external:
            try: 
                return self._read()
            except ValueError as e:
                raise
        features = []
        img_names = []
        for feature_path in self.extracted_faces_embeddings_path.glob("*.npy"):
            features.append(np.load(feature_path))

            # get the reference img of the face
            img_ext = Path(feature_path.stem).suffix
            img_name = feature_path.stem.split("_face")[0]
            img_names.append(img_name + img_ext)

        features = np.array(features, dtype=object).astype(float)
        self._write(features, img_names)
        return features, img_names

    def _add_to_embedding_store(self, query_feature: np.ndarray, query_img_path: str):
        try:
            self._append(query_feature, query_img_path)
        except ValueError as e:
            logger.error(str(e))

    def mark_as_processed(self, query_feature: np.ndarray, query_img_path: str, query_face_paths: List[str] ):
        store_in_redis(query_img_path, query_face_paths)

        #store in embedding_store
        from threading import Thread
        thread = Thread(target = self._add_to_embedding_store, args=(query_feature, query_img_path))   
        thread.start() 

class FaissEmbeddingsStore:
    def __init__(self, database):
        self.database = database
        self._faiss_store = self.database / "faiss_index.bin"
        self._names_store = self.database / "faiss_names.pkl"
        
        # Paths initialized via ImageProcessor
        self.img_data = self.database / "img_repo" / "img_data"
        self.extracted_faces_embeddings_path = self.database / "img_repo" / "extracted_faces_embeddings"
        
        self._lock = threading.Lock()
        self.index = None
        self.img_names = []
        self.dim = None


    def _load_index_in_mem(self):
        """Load existing FAISS index and names if available."""
        if not (self._faiss_store.exists() and self._names_store.exists()):
            raise ValueError("Embedding store doesn't exist")
        try:
            self.index = faiss.read_index(str(self._faiss_store))
            self.dim = self.index.d
            with open(self._names_store, "rb") as f:
                self.img_names = pickle.load(f)
            logger.info(f"Loaded FAISS index with {self.index.ntotal} embeddings")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            self.index = None
            self.img_names = []

    def _save_index(self):
        """Save FAISS index and names to disk."""
        if self.index is not None:
            faiss.write_index(self.index, str(self._faiss_store))
            with open(self._names_store, "wb") as f:
                pickle.dump(self.img_names, f)

    def _check_store_in_mem(self):
        """Ensure FAISS index and names are loaded in memory."""
        if not self.index or self.index.ntotal == 0 or not self.img_names:
            logger.warning("Embedding store not loaded; loading now...")
            self._load_index_in_mem()
        else:
            logger.info("Embedding store already loaded in memory.")

    def _read(self) -> Tuple[np.ndarray, List[str]]:
        """Read embeddings from FAISS index."""
        self._check_store_in_mem()
        logger.info("Reading in progress...")
        features = np.zeros((self.index.ntotal, self.index.d), dtype=np.float32)
        self.index.reconstruct_n(0, self.index.ntotal, features)
        return features, self.img_names.copy()

    def _write(self, features: np.ndarray, img_names: List[str]):
        """Write embeddings to FAISS index."""
        with self._lock:
            features = features.astype(np.float32)
            
            # Normalize for cosine similarity
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)  # Avoid division by zero
            features = features / norms

            self.dim = features.shape[1]
            # Use IndexFlatIP for cosine similarity on normalized vectors
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(features)
            self.img_names = img_names.copy()
            self._save_index()
            logger.info(f"Written {len(img_names)} embeddings to FAISS index")

    @staticmethod
    def _l2_normalize(query_feature: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length for cosine similarity."""
        norms = np.linalg.norm(query_feature, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # Avoid division by zero
        query_feature = query_feature / norms
        return query_feature

    def get_similar_faces(self, query_feature: np.ndarray, threshold: float) -> List[Tuple[float, str]]:
        """Get similar faces using FAISS for fast similarity search."""
        # Auto-load embeddings if index is empty
        if self.index is None or self.index.ntotal == 0:
            logger.info("Index empty, loading embeddings from files...")
            self.load_all_embeddings_in_faiss()
        
        try:
            # CRITICAL: Ensure 2D array and proper dtype
            query_feature = np.atleast_2d(np.array(query_feature, dtype=np.float32))
            query_feature = FaissEmbeddingsStore._l2_normalize(query_feature)
            
            # Use range_search for threshold-based search
            # lims, similarities, indices = self.index.range_search(query_feature, threshold)
            
            results = []
            
            # # Process all queries (automatically handles single or multiple)
            # for i in range(query_feature.shape[0]):
            #     start, end = lims[i], lims[i + 1]
            #     for sim, idx in zip(similarities[start:end], indices[start:end]):
            #         if idx < len(self.img_names):
            #             results.append((float(sim), self.img_names[int(idx)]))
            
            #  Process queries one by one to avoid memory explosion
            for i in range(query_feature.shape[0]):
                single_query = query_feature[i:i+1]  # Shape: (1, dim)
                
                # Use range_search for single query only
                lims, similarities, indices = self.index.range_search(single_query, threshold)
                
                # Process results for this single query
                start, end = lims[0], lims[1]
                for sim, idx in zip(similarities[start:end], indices[start:end]):
                    if idx < len(self.img_names):
                        results.append((float(sim), self.img_names[int(idx)]))
                    
            # Sort by similarity (descending)
            results.sort(key=lambda x: x[0], reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Error in FAISS range search: {e}")
            logger.error(f"Query feature shape: {query_feature.shape if 'query_feature' in locals() else 'undefined'}")
            logger.error(f"Query feature type: {type(query_feature) if 'query_feature' in locals() else 'undefined'}")
            return []

    def search_topk(self, query_feature: np.ndarray, k: int = 50) -> List[Tuple[float, str]]:
        """Search for top-k most similar embeddings."""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is empty")
            return []

        # Normalize query feature for cosine similarity
        query_feature = FaissEmbeddingsStore._l2_normalize(np.array(query_feature, dtype=np.float32).reshape(1, -1))
        
        search_k = min(k, self.index.ntotal)
        similarities, indices = self.index.search(query_feature, search_k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != -1 and idx < len(self.img_names):
                results.append((float(sim), self.img_names[int(idx)]))
        
        return results
    
    def load_all_embeddings_in_faiss(self, sync_mode: bool = False, external:bool=False):
        """Load all face embeddings into FAISS index."""
    
        def _worker():
            if external:
                self._load_index_in_mem()
                return
            features = []
            img_names = []
            
            for feature_path in self.extracted_faces_embeddings_path.glob("*.npy"):
                features.append(np.load(feature_path))
                
                # Get the reference img of the face
                img_ext = Path(feature_path.stem).suffix
                img_name = feature_path.stem.split("_face")[0]
                img_names.append(img_name + img_ext)

            if not features:
                logger.warning("No face embeddings found")
                return np.empty((0, 512), dtype=np.float32), []

            features = np.array(features, dtype=object).astype(np.float32)
            
            self._write(features, img_names)  # persists + rebuilds FAISS
            logger.info(f"Loaded {len(img_names)} embeddings into FAISS index.")

        if sync_mode:
            from threading import Thread
            t = Thread(target=_worker, daemon=True)
            t.start()
            logger.info("Started background FAISS embeddings load.")
        else:
            _worker()
        #return features, img_names

    def _add_and_rebuild_index(self, query_feature: np.ndarray, query_img_path: str):
        """Core: add `query_feature` and rebuild the index. Must be called under lock."""
        try: 
            query_img_name = str(Path(query_img_path).name)

            # Skip if already in store
            if query_img_name in self.img_names:
                logger.info(f"{query_img_name} already exists in embedding store. Skipping.")
                return
            
            try:
                existing_features, existing_names = self._read() 
            except ValueError:
                # Store not initialized yet
                existing_features = np.empty((0, query_feature.shape[1]), dtype=np.float32)
                existing_names = []

            # Build combined matrix + names
            combined_features = query_feature if existing_features.size == 0 else np.vstack([existing_features, query_feature])
            updated_img_names = existing_names.append(query_img_name)   
            
            self._write(combined_features, updated_img_names)   # <- your existing writer that rebuilds FAISS + saves names
            logger.info(f"SUCCESS: {query_img_name} added and index rebuilt with {len(updated_img_names)} embeddings.")
        except ValueError as v:
            logger.error(f"Error adding to index: {e}")
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            
    def add_feature(self, query_feature: np.ndarray, query_img_path: str, sync_mode:bool=True):
        """rebuild sync or in background thread."""
        from threading import Thread
        def _worker():
            with self._lock:
                self._add_and_rebuild_index(query_feature, query_img_path)
                
        if sync_mode:        
            t = Thread(target=_worker, daemon=True)
            t.start()
            logger.info(f"Started background rebuild for {Path(query_img_path).name}")
        else:
            _worker()
# Global instance
embeddings_handler = FaissEmbeddingsStore(database)
embeddings_handler.load_all_embeddings_in_faiss(external=True)