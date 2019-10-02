import os
import shutil
import numpy as np 
from PIL import Image
import random
from skimage.transform import downscale_local_mean, resize

from typing import Tuple

def rotate_images(image_directory: str) -> None:
    images = [os.path.join(image_directory, f) for f in os.listdir(image_directory)]
    for image in images:

        image_file_split = image.split(".")
        image_file_name = image_file_split[0]
        image_file_extension = image_file_split[1]
        
        img = Image.open(image)

        img.rotate(90).save("{}r90.{}".format(image_file_name, image_file_extension))
        img.rotate(180).save("{}r180.{}".format(image_file_name, image_file_extension))
        img.rotate(270).save("{}r270.{}".format(image_file_name, image_file_extension))

        img.close()

def downsample_images(image_directory: str, output_shape: Tuple[int]) -> None:
    images = [os.path.join(image_directory, f) for f in os.listdir(image_directory)]
    downsampled_images = []
    for image in images:
        img = np.array(Image.open(image), dtype=float)
        downscaled_image = resize(img, output_shape, anti_aliasing=True)
        downsampled_images.append(downscaled_image)
    return downsampled_images

def split_test_train(image_directory: str, n_train, train_destination:str, test_destination:str):
    languages = os.listdir(image_directory)
    language_directories = [os.path.join(image_directory, language) for language in languages]

    character_directories = np.concatenate([[os.path.join(language_dir, character_dir) for character_dir in os.listdir(language_dir)] for language_dir in language_directories])
    
    train_data_ids = random.sample(list(range(len(character_directories))), n_train) 
    
    train_data = [character_directories[i] for i in train_data_ids]
    test_data = [d for d in character_directories if d not in train_data]

    for c, char_data in enumerate(train_data):
        shutil.copytree(char_data, "{}/char{}".format(train_destination, c))
    for c, char_data in enumerate(test_data):
        shutil.copytree(char_data, "{}/char{}".format(test_destination, c))
