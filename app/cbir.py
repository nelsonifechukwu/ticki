from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from deepface import DeepFace
from deepface.basemodels import VGGFace
from retinaface import RetinaFace
import os
from PIL import Image

# See https://keras.io/api/applications/ for details

class ImageProcessor:
    
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input,
                    outputs=base_model.get_layer('fc1').output)
    @staticmethod    
    def extract_faces(database: Path):
        """
        Extract faces from all images in a directory
        Args:
            database: path to the directory containing images
        """
        if not isinstance(database, Path):
            raise TypeError("The 'database' parameter must be a Path object.")
        if not database.exists():
            raise FileNotFoundError(f"The directory '{database}' does not exist.")
    
        faces_directory = database / "faces"
        faces_directory.mkdir(parents=True, exist_ok=True)
                        
        all_pics = Path(database).iterdir()
        for pic in all_pics:
            if pic.is_file() and pic.suffix.lower() in ['.png', '.jpg']:
                # Load the image
                img = mpimg.imread(pic)
                # plt.imshow(img)
                # plt.show(block=False)
                # plt.pause(10)
                faces = RetinaFace.extract_faces(img_path=str(pic), align=True, expand_face_area=20)
                for i, face in enumerate(faces):
                    if face.any():
                        img = Image.fromarray(face)
                        face_filename = f"{pic.stem}_{i}.png"
                        face_filepath = faces_directory / face_filename
                        img.save(face_filepath)
    
    @staticmethod          
    def feature_extractor(faces: Path):
        """
        Extract features from a directory of faces
        Args:
            faces: path
        """
        path=faces
        for face in path.iterdir():       
             # Convert the image to a NumPy array
            if face.suffix != ".npy":
                img = Image.open(face)
                img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
                img = img.convert('RGB')  # Make sure img is color
                
                # To np.array. Height x Width x Channel. dtype=float32
                x = image.img_to_array(img)
                
                # (H, W, C)->(1, H, W, C), where the first elem is the number of img
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)  # Subtracting avg values for each pixel
                
                # Get the feature vector
                feature = ImageProcessor.model.predict(x)[0]  # (1, 4096) -> (4096, )
                image_embedding = feature / np.linalg.norm(feature)  # Normalize

                # Save the array to a .npy file
                np.save(f"{path}/{face.stem}" + ".npy", image_embedding)


# def compare_pics(database:str, test_input:str):
#     all_pics = os.listdir(database)
#     for i, pic in enumerate(all_pics):
#         location = Path.cwd() / f"{database}" / f"{pic}"
#         faces = RetinaFace.extract_faces(img_path=location, align=True, expand_face_area=20)
#         for  face in faces:
#             if face.any():
#                 img = Image.fromarray(face)
#                 img.save('f.png')
#                 result = DeepFace.verify('f.png',
#                             test_input, model_name="ArcFace", normalization="ArcFace", distance_metric="euclidean_l2", detector_backend="mtcnn", enforce_detection=False)
#                 print(test_input, location, result["verified"])
#                 plt.imshow(face)
#                 plt.show(block=False)
#                 plt.pause(1)
#                 plt.close()
# # compare_pics('girl', 'girl/test.JPG')

# def showwith_retina():
#     location = "a2342e6d-a248-497a-ae02-5eacd85213be.JPG"
#     faces = RetinaFace.detect_faces(img_path=location)
#     r_img = cv.imread(location)
#     for face in faces:
#         place = faces[f"{face}"][f"facial_area"]
#         pt1 = (place[0], place[1])
#         pt2 = (place[2], place[3])
#         cv.rectangle(r_img, pt1, pt2, color=(0, 255, 0), thickness=2)
#     plt.imshow(r_img)
#     plt.show()
