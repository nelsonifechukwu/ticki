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
    
    def _check_store(self):
        if not self._store.exists():
            raise ValueError("No external embedding store available")
    def _read(self) -> Tuple[np.ndarray, List[str]]:
        self._check_store()
        with h5py.File(self._store, 'r') as file:
                features = file['embeddings'][:]
                img_paths = [path.decode('utf-8') for path in file['img_paths'][:]]
        return features, img_paths
    
    def _write(self, features: np.ndarray, img_paths: List[str]):
        with h5py.File(self._store, 'w') as file:
            file.create_dataset('embeddings', data=features)
            dt = h5py.string_dtype(encoding='utf-8')
            file.create_dataset('img_paths', data=img_paths, dtype=dt)
    
    def append(self, query_feature, query_img_path: Union[Path, str]):
        features, img_paths = self._read()
        query_img_path_str = str(query_img_path)
      # Skip if already in store
        if query_img_path_str in img_paths:
            print(f"{query_img_path_str} already exists in embedding store. Skipping append.")
            return
        features = np.vstack([features, query_feature])    
        img_paths.append(query_img_path_str)
        self._write(features, img_paths)
        print(f"SUCCESS: {query_img_path_str} successfully added to embedding store.")
    
    def load_allfaces_embeddings(self, external=None) -> Tuple[np.ndarray, List[str]]: 
        #load external embeddings
        if external:
            try: 
                return self._read()
            except ValueError as e:
                raise
  
        features = []
        img_paths = []
        base_path = Path("app/static")
        for feature_path in self.extracted_faces_embeddings_path.glob("*.npy"):
            features.append(np.load(feature_path))
            # From 'IMG_3011_face_0.JPG.npy' to
            img_ext = Path(feature_path.stem).suffix  # '.JPG'
            img_name = feature_path.stem.split("_face")[0]  # 'IMG_3011'
            img_paths.append(self.img_data.relative_to(base_path) / (img_name + img_ext)) #get the reference img of the face

        img_paths = [str(path) for path in img_paths]
        features = np.array(features, dtype=object).astype(float)
        self._write(features, img_paths)
        return features, img_paths
    
    def _add_to_embedding_store(self, query_img_path, query_feature):
        base_path = Path("app/static")
        query_img_path = query_img_path.relative_to(base_path)
        try:
            self.append(query_feature, query_img_path)
        except ValueError as e:
            print(e)
            
    def bg_store(self, query_feature: np.ndarray, query_img_path: str,query_face_paths: List[str] ):
        from threading import Thread
        #check if there's any file in the upload folder 
        store_in_redis(query_img_path, query_face_paths)
        thread = Thread(target = self._add_to_embedding_store, args=(query_img_path, query_feature,))
        #thread.daemon = True  # Dies with the main thread
        thread.start()

embeddings_store = EmbeddingsStore(database)


# try:   
#     # all_face_embeddings, all_face_paths = None
# all_face_embeddings, all_face_paths = embeddings_store.load_allfaces_embeddings(external=True)
# except ValueError as e:
#     all_face_embeddings = all_face_paths = None
#     print(e)
    