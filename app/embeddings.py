import threading
import h5py
from typing import Tuple, List, Union
import numpy as np
from pathlib import Path
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
            print(f"{query_img_name} already exists in embedding store. Skipping append.")
            return
        features = np.vstack([features, query_feature])    
        img_names.append(query_img_name)
        self._write(features, img_names)
        print(f"SUCCESS: {query_img_name} successfully added to embedding store.")
    
    def load_allfaces_embeddings(self, external=None) -> Tuple[np.ndarray, List[str]]: 
        #load external embeddings
        if external:
            try: 
                return self._read()
            except ValueError as e:
                raise
        #with self._lock:
        features = []
        img_names = []
        for feature_path in self.extracted_faces_embeddings_path.glob("*.npy"):
            features.append(np.load(feature_path))
            
            # get the reference img of the face
            # From 'IMG_3011_face_0.JPG.npy' to
            img_ext = Path(feature_path.stem).suffix  # '.JPG' to
            img_name = feature_path.stem.split("_face")[0]  # 'IMG_3011' to
            img_names.append(img_name + img_ext) # 'IMG_3011.JPG'

        features = np.array(features, dtype=object).astype(float)
        self._write(features, img_names)
        return features, img_names
    
    def _add_to_embedding_store(self, query_feature: np.ndarray, query_img_path: str):
        try:
            self._append(query_feature, query_img_path)
        except ValueError as e:
            print(e)
            
    def mark_as_processed(self, query_feature: np.ndarray, query_img_path: str, query_face_paths: List[str] ):
        store_in_redis(query_img_path, query_face_paths)

        #store in embedding_store
        from threading import Thread
        thread = Thread(target = self._add_to_embedding_store, args=(query_feature, query_img_path))   
        thread.start() 

embeddings_handler = EmbeddingsStore(database)


# try:   
#     # all_face_embeddings, all_face_paths = None
# all_face_embeddings, all_face_paths = embeddings_store.load_allfaces_embeddings(external=True)
# except ValueError as e:
#     all_face_embeddings = all_face_paths = None
#     print(e)
    