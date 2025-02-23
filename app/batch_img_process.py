# from .cbir import ImageProcessor
# from pathlib import Path
# import threading
# from concurrent.futures import ThreadPoolExecutor

# thread_local = threading.local()
# exe = ImageProcessor()

# database = Path("app/static/database")
# img_repo = database / "img_repo"
# img_repo_list = list(img_repo.iterdir())
# def convert_all_faces_to_embeddings(repository: Path):
#     #extract all faces
#     with ThreadPoolExecutor(max_workers = 10) as executor:
#         executor.map(exe.extract_faces_thread, repository)

# convert_all_faces_to_embeddings(img_repo_list)