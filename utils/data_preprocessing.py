import os
import shutil
import numpy as np 
from PIL import Image
import random
from skimage.transform import downscale_local_mean, resize

from typing import Tuple

def rotate_images(image_directory: str) -> None:
    """
    For all images in directory provided, make 3 copies of each image, rotated 90, 180 and 270 degrees respectively

    :param image_directory: path of images to rotate
    """
    images = [os.path.join(image_directory, f) for f in os.listdir(image_directory)]

    if images[0].endswith('png'):

        for image in images:

            image_file_split = image.split(".")
            image_file_name = image_file_split[0]
            image_file_extension = image_file_split[1]
            
            img = Image.open(image)

            img.rotate(90).save("{}r90.{}".format(image_file_name, image_file_extension))
            img.rotate(180).save("{}r180.{}".format(image_file_name, image_file_extension))
            img.rotate(270).save("{}r270.{}".format(image_file_name, image_file_extension))

            img.close()
    
    elif images[0].endswith('npy'):

        for image in images:

            image_file_split = image.split(".")
            image_file_name = image_file_split[0]
            
            img90 = np.rot90(np.load(image))
            img180 = np.rot90(img90)
            img270 = np.rot90(img180)

            np.save("{}r90".format(image_file_name), img90)
            np.save("{}r180".format(image_file_name), img180)
            np.save("{}r270".format(image_file_name), img270)

    else:
        raise ValueError("Unknown file type for image data")

def downsample_images(image_directory: str, output_shape: Tuple[int], save_as_array: bool=True) -> None:
    """
    Reduce size of images in directory provided to a size specified.

    :param image_directory: path to images to be downsampled
    :param output_shape: size of images desired
    """
    images = [os.path.join(image_directory, f) for f in os.listdir(image_directory)]
    for image in images:
        img = np.array(Image.open(image), dtype=float)
        downsampled_image = resize(img, output_shape, anti_aliasing=True)
        if save_as_array:
            np.save(image.split(".")[0], downsampled_image)
            os.remove(image)
        else:
            downsampled_image = Image.fromarray(np.round(resize(img, output_shape, anti_aliasing=True)))
            downsampled_image.convert('L').save(image)

def split_test_train(image_directory: str, n_train:int, train_destination:str, test_destination:str):
    """
    Split training and test data

    :param image_directory: path to image data
    :param n_train: number of characters to use for training
    :param train_destination: path to save training data
    :param test_destination: path to save test data
    """
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

def preprocess_images(image_directory:str, n_train: int, output_shape: Tuple[int]):
    """
    Perform following preprocessing steps:

    - rotate all images in dataset through 90 degrees for data augmentation
    - downsample all images to 28 x 28
    - split data (regardless of alphabet) into training/test set specified by configuration

    :param image_directory: path to image data
    :param n_train: number of characters to use for training
    :param train_destination: path to save training data
    :param test_destination: path to save test data
    :param output_shape: shape of downsampled image
    """
    image_path_split = image_directory.split('/')
    if len(image_path_split[-1]) > 0:
        image_folder_name_length = len(image_path_split[-1]) 
    else:
        image_folder_name_length = len(image_path_split[-2]) + 1
    data_dir = image_directory[:-image_folder_name_length]
    train_destination = os.path.join(data_dir, 'train_data')
    test_destination = os.path.join(data_dir, 'test_data')

    split_test_train(image_directory, n_train=n_train, train_destination=train_destination, test_destination=test_destination)

    training_character_paths = [os.path.join(train_destination, char_path) for char_path in os.listdir(train_destination)]
    test_character_paths = [os.path.join(test_destination, char_path) for char_path in os.listdir(test_destination)]

    for train_char_path in training_character_paths:
        downsample_images(train_char_path, output_shape=output_shape)
    for test_char_path in test_character_paths:
        downsample_images(test_char_path, output_shape=output_shape)
        rotate_images(test_char_path)

    print()
