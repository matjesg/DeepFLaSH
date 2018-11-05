"""
U-Net
Common utility functions and classes.
Licensed under the MIT License (see LICENSE for details)
Written by Matthias Griebel
"""

import os
import requests
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.optimizers import Adam
from unet.metrics import recall, precision, f1, mcor
from unet.losses import weighted_bce_dice_loss
from scipy.misc import imsave
from scipy.spatial.distance import jaccard
import numpy as np
import matplotlib.pyplot as plt
from unet.model import unet_1024

############################################################
#  Google Drive Download
#  from https://stackoverflow.com/a/39225039
#  and https://github.com/nsadawi/Download-Large-File-From-Google-Drive-Using-Python/blob/master/Download-Large-File-from-Google-Drive.ipynb
############################################################

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

############################################################
#  Read and Resize Images
############################################################
def readImg(img_name, path, channels, dimensions):
    if channels == 1:
        grayscale = 1
    else:
        grayscale = 0
    img = load_img(os.path.join(path, img_name), grayscale=grayscale, target_size = [dimensions, dimensions])
    img = img_to_array(img)
    img /= 255

    return (img)

############################################################
#  Save Masks
############################################################
def saveMasks(msk_list, img_names, filetype = 'tif'):
    path = 'masks'
    if not os.path.isdir(path):
        os.makedirs(path)
    for i in range(len(img_names)):
        outfile = os.path.join(path, img_names[i] + '_mask.' + filetype)
        imsave(outfile, np.squeeze(msk_list[i], axis=2))

############################################################
#  Load U-net Model
############################################################

def load_unet(model_name):

    if model_name == 'new':
        model = unet_1024(img_rows=1024,
                          img_cols=1024,
                          num_img_channels=1,
                          num_mask_channels=1)

        model.compile(loss = weighted_bce_dice_loss(),
                      optimizer = Adam(lr=0.001),
                      metrics=['accuracy',
                               recall,
                               precision,
                               f1,
                               mcor])
    else:
        path = 'saved_models'
        id = ''
        if not os.path.isdir(path):
            os.makedirs(path)

        file_path = os.path.join(path, model_name + '_unet.h5')

        if not os.path.isfile(file_path):
            if model_name == 'cFOS':
                id = '1f8KhLuvwCVmxBxECMSUjLG5slczBzBhj'
            elif model_name == 'Parv':
                id = '1lXsUUJWbjk86ZX6IQQaKH7gAwUO_nlOV'
            else:
                print('Please provide correct unet name (cFOS, Parv or new)')
                return

            print('ID: ', id)
            print('Download file to ', file_path)

            download_file_from_google_drive(id, file_path)

        model = load_model(file_path,
                           custom_objects={'recall': recall,
                                            'precision': precision,
                                            'f1': f1,
                                            'mcor': mcor,
                                            'weighted_bce_dice_loss': weighted_bce_dice_loss})
    return(model)

############################################################
#  Compare Masks to Expert using the Jaccard Similarity
############################################################

def jaccard_sim(list1, list2, threshold = 0.5):
    list1 = [(map > threshold).astype(np.uint8) for map in list1]
    list2 = [(map > threshold).astype(np.uint8) for map in list2]
    jac = [jaccard(list1[i].flatten(), list2[i].flatten()) for i in range(len(list1))]
    return(jac)

############################################################
#  Join Mask Results for Plotting
############################################################

def join_masks(msk, exp, threshold = 0.5):
    msk = (msk > threshold).astype(np.uint8)
    join = np.append(msk, exp, axis=2)
    join = np.append(join, msk, axis=2)
    return(join)

############################################################
#  Plot images and mask
############################################################

def plot_image_and_mask(img_names, img_list,
                        msk_names, msk_list,
                        img_head='Image', msk_head='Mask'):
    plt.tight_layout()
    for i in range(len(img_names)):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))

        if img_list[i].shape[2] == 1:
            img = np.squeeze(img_list[i], axis=2) * (-1)
        else:
            img = img_list[i]

        if msk_list[i].shape[2] == 1:
            msk = np.squeeze(msk_list[i], axis=2) * (-1)
        else:
            msk = msk_list[i]

        axs[0].imshow(img)
        axs[0].set_title(str(img_head + ': ' + str(img_names[i])))
        axs[0].axis('off')
        axs[1].imshow(msk)
        axs[1].set_title(str(msk_head + ': ' + str(msk_names[i])))
        axs[1].axis('off')
        fig.show()

############################################################
#  Create Data Generator
############################################################

def create_generator(img_list, msk_list, SEED=1, BATCH_SIZE=4):

    data_gen_args = dict(rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         shear_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode="constant",
                         cval=0)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    image_generator = image_datagen.flow(np.asarray(img_list),
                                         seed=SEED, batch_size=BATCH_SIZE)
    mask_generator = mask_datagen.flow(np.asarray(msk_list),
                                        seed=SEED, batch_size=BATCH_SIZE)

    return (zip(image_generator, mask_generator))