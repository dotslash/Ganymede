from typing import List

import numpy as np
from keras import Model
from keras.models import load_model
import cv2
from keras_preprocessing import image
from keras_vggface.utils import preprocess_input


# def load_image_as_array(filepath: str) -> np.ndarray:
#     with open(filepath, 'rb') as f:
#         # noinspection PyTypeChecker
#         nparr = np.fromstring(f.read(), np.uint8)
#         # noinspection PyUnresolvedReferences
#         return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def load_image_as_array(filepath: str) -> np.ndarray:
    img = image.load_img(filepath, target_size=(224, 224))
    img = np.array(img).astype(np.float)
    return preprocess_input(img, version=2)

MODEL_FILES = ['/home/stps/tmp/kinship_0.m5', '/home/stps/tmp/kinship_1.m5',
               '/home/stps/tmp/kinship_2.m5', '/home/stps/tmp/kinship_3.m5',
               '/home/stps/tmp/kinship_4.m5']
# IMAGES_DIR = '/home/stps/Downloads/kinship_images'
# import os
# images_files = os.listdir(IMAGES_DIR)
# images = [load_image_as_array(images_file) for images_file in images_files]
# for i in range(len(images)):
#     for j in range(len(images)):
#         print(i,j)




i1: np.ndarray = load_image_as_array('/home/stps/Downloads/teja1.jpg')
i2: np.ndarray = load_image_as_array('/home/stps/Downloads/harini.jpg')
print(i1.shape, i2.shape)

models: List[Model] = []
for model_file in MODEL_FILES:
    model: Model = load_model(model_file)
    models.append(model)
