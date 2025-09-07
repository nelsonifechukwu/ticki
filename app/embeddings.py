import h5py
import faiss
import threading
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, List, Literal
from .cbir import logger
from .tasks import database #store_in_redis


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
    def __init__(self, database, index_type: Literal["flat", "ivf"] = "flat"):
        self.database = database
        self.index_type = index_type
        
        if index_type=="flat":
            self._faiss_store = self.database / "faiss_index.bin"
            self._names_store = self.database / "faiss_names.pkl"
        elif index_type=="ivf":
            self._faiss_store = self.database / "ivf_faiss_index.bin"
            self._names_store = self.database / "ivf_faiss_names.pkl"
        
        # Paths initialized via ImageProcessor
        self.img_data = self.database / "img_repo" / "img_data"
        self.extracted_faces_embeddings_path = self.database / "img_repo" / "extracted_faces_embeddings"
        
        self._lock = threading.Lock()
        self.index = None
        self.img_names = []
        self.dim = None

        logger.info(f"Initialized FAISS store with {self.index_type.upper()} index type")

    def _load_index_in_mem(self):
        """Load existing FAISS index and names if available."""
        if not (self._faiss_store.exists() and self._names_store.exists()):
            raise ValueError("Embedding store doesn't exist")
        try:
            self.index = faiss.read_index(str(self._faiss_store))
            self.dim = self.index.d
            with open(self._names_store, "rb") as f:
                self.img_names = pickle.load(f)
            logger.info(f"Loaded FAISS {self.index_type.upper()} index with {self.index.ntotal} embeddings")
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
    
    def _create_index(self, dim: int, num_vectors: int) -> faiss.Index:
        """Create appropriate FAISS index based on index_type."""
        if self.index_type == "flat":
            logger.info(f"Creating IndexFlatIP with dimension {dim}")
            return faiss.IndexFlatIP(dim)
        
        elif self.index_type == "ivf":
            # Adaptive nlist based on data size
            nlist = min(100, max(1, num_vectors // 10))
            logger.info(f"Creating IndexIVFFlat with dimension {dim} and {nlist} clusters")
            quantizer = faiss.IndexFlatIP(dim)
            return faiss.IndexIVFFlat(quantizer, dim, nlist)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

    def _write(self, features: np.ndarray, img_names: List[str]):
        """Write embeddings to FAISS index."""
        with self._lock:
            features = features.astype(np.float32)
            
            # Normalize for cosine similarity
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)  # Avoid division by zero
            features = features / norms

            self.dim = features.shape[1]
            num_vectors = features.shape[0]
            
            self.index = self._create_index(self.dim, num_vectors)
            # Train index if needed (IVF requires training)
            if self.index_type == "ivf":
                nlist = self.index.nlist
                if num_vectors >= nlist:
                    logger.info(f"Training IVF index with {num_vectors} vectors and {nlist} clusters...")
                    self.index.train(features)
                    logger.info("Training complete.")
                else:
                    logger.warning(f"Not enough vectors ({num_vectors}) to train {nlist} clusters. Using minimum clusters.")
                    nlist = max(1, num_vectors)
                    quantizer = faiss.IndexFlatIP(self.dim)
                    self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
                    self.index.train(features)
            
            # Add vectors to index
            self.index.add(features)
            self.img_names = img_names.copy()
            self._save_index()
            
            index_info = f"with {self.index.nlist} clusters" if self.index_type == "ivf" else ""
            logger.info(f"Written {len(img_names)} embeddings to FAISS {self.index_type.upper()} index {index_info}")

    @staticmethod
    def _l2_normalize(query_feature: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length for cosine similarity."""
        norms = np.linalg.norm(query_feature, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # Avoid division by zero
        query_feature = query_feature / norms
        return query_feature

    def get_similar_faces(self, query_feature: np.ndarray, threshold: float=0) -> List[Tuple[float, str]]:
        """Get similar faces using FAISS for fast similarity search."""
        # Auto-load embeddings if index is empty
        if self.index is None or self.index.ntotal == 0:
            logger.info("Index empty, loading embeddings from files...")
            self.load_all_embeddings_in_faiss()
        
        try:
            # Ensure 2D array and proper dtype
            query_feature = np.atleast_2d(np.array(query_feature, dtype=np.float32))
            query_feature = FaissEmbeddingsStore._l2_normalize(query_feature)
            
            results = []
            # # Process all queries (automatically handles single or multiple)
            # Use range_search for threshold-based search
            # lims, similarities, indices = self.index.range_search(query_feature, threshold)

            # for i in range(query_feature.shape[0]):
            #     start, end = lims[i], lims[i + 1]
            #     for sim, idx in zip(similarities[start:end], indices[start:end]):
            #         if idx < len(self.img_names):
            #             results.append((float(sim), self.img_names[int(idx)]))
            
            #  Process queries one by one to avoid memory explosion
            
            for i in range(query_feature.shape[0]):
                single_query = query_feature[i:i+1]  # Shape: (1, dim)
                
                if self.index_type == "flat":
                    # Use range_search for exact threshold matching (FLAT only)
                    lims, similarities, indices = self.index.range_search(single_query, threshold)
                    start, end = lims[0], lims[1]
                    for sim, idx in zip(similarities[start:end], indices[start:end]):
                        if idx < len(self.img_names):
                            results.append((float(sim), self.img_names[int(idx)]))
                
                elif self.index_type == "ivf":
                    # Use top-k search then filter by threshold (IVF)
                    k = min(1000, self.index.ntotal)  # Search more candidates since it's approximate
                    similarities, indices = self.index.search(single_query, k)
                    for sim, idx in zip(similarities[0], indices[0]):
                        if idx != -1 and idx < len(self.img_names) and sim >= threshold:
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
    
    def load_all_embeddings_from_npy_files(self, cleanup_npy: bool = True):
        """Load all .npy face embeddings into FAISS index and optionally clean up files."""
        try:
            features = []
            face_ids = []
            
            npy_files = list(self.extracted_faces_embeddings_path.glob("*.npy"))
            if not npy_files:
                logger.warning("No face embeddings (.npy files) found")
                return
            
            logger.info(f"Loading {len(npy_files)} .npy embedding files...")
            
            for feature_path in npy_files:
                try:
                    embedding = np.load(feature_path)
                    features.append(embedding)
                    
                    # Use the full filename (with _face_i) as the face ID
                    face_id = feature_path.stem  # e.g., "IMG_001_face_0"
                    face_ids.append(face_id.split("_face")[0]) #id=IMG_001 (only name)
                    
                except Exception as e:
                    logger.error(f"Failed to load {feature_path}: {e}")

            if not features:
                logger.warning("No valid embeddings loaded from .npy files")
                return

            # Stack all embeddings and write to FAISS
            features_array = np.vstack(features).astype(np.float32)
            self._write(features_array, face_ids)
            
            logger.info(f"✅ Loaded {len(face_ids)} embeddings into FAISS index")
            
            # Clean up .npy files after successful load
            if cleanup_npy:
                for npy_file in npy_files:
                    try:
                        npy_file.unlink()
                    except Exception as e:
                        logger.error(f"Failed to delete {npy_file}: {e}")
                logger.info(f"🧹 Cleaned up {len(npy_files)} .npy files")
                        
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings from .npy files: {e}")
            raise


    def add_feature(self, query_feature: np.ndarray, query_img_path, sync_mode: bool = True):
        """
        Add embeddings and rebuild index.
        query_feature: np.ndarray of shape (1, dim) for single or (n, dim) for batch
        query_img_path: str for single or List[str] for batch
        """
        from threading import Thread
        
        def _worker():

            try:
                # Ensure query_feature is 2D numpy array
                if isinstance(query_feature, list):
                    features = np.vstack(query_feature).astype(np.float32)
                else:
                    features = np.atleast_2d(query_feature).astype(np.float32)
                
                # Handle face_ids
                if isinstance(query_img_path, list):
                    face_ids = [str(Path(path).name) if isinstance(path, str) else path for path in query_img_path]
                else:
                    face_ids = [str(Path(query_img_path).name) if isinstance(query_img_path, str) else query_img_path]
                
                # Validate dimensions
                if len(face_ids) != features.shape[0]:
                    raise ValueError(f"Mismatch: {features.shape[0]} embeddings but {len(face_ids)} face_ids")
                
                # Filter out duplicates
                new_features = []
                new_ids = []
                for i, face_id in enumerate(face_ids):
                    if face_id not in self.img_names:
                        new_features.append(features[i])
                        new_ids.append(face_id)
                    else:
                        logger.info(f"{face_id} already exists in embedding store. Skipping.")
                
                if not new_features:
                    logger.info("All embeddings already exist in store. Skipping.")
                    return
                
                # Stack new features
                new_features_array = np.vstack(new_features)
                
                # Combine with existing embeddings
                try:
                    existing_features, existing_names = self._read()
                    combined_features = np.vstack([existing_features, new_features_array])
                    combined_names = existing_names + new_ids
                except ValueError:
                    # Store not initialized yet
                    combined_features = new_features_array
                    combined_names = new_ids
                
                # Rebuild index with all embeddings
                self._write(combined_features, combined_names)
                logger.info(f"✅ Added {len(new_ids)} embeddings. Total in index: {len(combined_names)}")
                
            except Exception as e:
                logger.error(f"❌ Error adding embeddings: {e}")
                raise
        
        if sync_mode:
            t = Thread(target=_worker, daemon=True)
            t.start()
            logger.info(f"Started background rebuild for {np.atleast_2d(query_feature).shape[0]} embeddings")
        else:
            _worker()
# Global instance
embeddings_handler = FaissEmbeddingsStore(database=database,index_type="flat")