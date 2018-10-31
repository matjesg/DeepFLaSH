"""
U-Net
The U-net unet implementation in Keras
Licensed under the MIT License (see LICENSE for details)
Written by Matthias Griebel
"""

from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.layers import UpSampling2D
from keras.layers import Activation
from keras.layers import SeparableConv2D
from keras import backend as K

############################################################
#  U-net
#  Adapted from
#  Ronneberger et al.: "U-net: Convolutional networks for biomedical image segmentation."
#  MICCAI. Springer, Cham, 2015.
############################################################
def unet_1024(img_rows, img_cols, num_img_channels, num_mask_channels, train_batch_norm = True):

    inputs = Input(shape=(img_rows, img_cols, num_img_channels))
    # 1024

    with K.name_scope('Down512'):
        down0b = Conv2D(8, (3, 3), padding='same')(inputs)
        down0b = BatchNormalization()(down0b, training = train_batch_norm)
        down0b = Activation('relu')(down0b)
        down0b = Conv2D(8, (3, 3), padding='same')(down0b)
        down0b = BatchNormalization()(down0b, training = train_batch_norm)
        down0b = Activation('relu')(down0b)
        down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    # 512

    with K.name_scope('Down256'):
        down0a = SeparableConv2D(16, (3, 3), padding='same')(down0b_pool)
        down0a = BatchNormalization()(down0a,training = train_batch_norm)
        down0a = Activation('relu')(down0a)
        down0a = SeparableConv2D(16, (3, 3), padding='same')(down0a)
        down0a = BatchNormalization()(down0a, training = train_batch_norm)
        down0a = Activation('relu')(down0a)
        down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    with K.name_scope('Down128'):
        down0 = SeparableConv2D(32, (3, 3), padding='same')(down0a_pool)
        down0 = BatchNormalization()(down0, training = train_batch_norm)
        down0 = Activation('relu')(down0)
        down0 = SeparableConv2D(32, (3, 3), padding='same')(down0)
        down0 = BatchNormalization()(down0, training = train_batch_norm)
        down0 = Activation('relu')(down0)
        down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    with K.name_scope('Down64'):
        down1 = SeparableConv2D(64, (3, 3), padding='same')(down0_pool)
        down1 = BatchNormalization()(down1, training = train_batch_norm)
        down1 = Activation('relu')(down1)
        down1 = SeparableConv2D(64, (3, 3), padding='same')(down1)
        down1 = BatchNormalization()(down1, training = train_batch_norm)
        down1 = Activation('relu')(down1)
        down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    with K.name_scope('Down32'):
        down2 = SeparableConv2D(128, (3, 3), padding='same')(down1_pool)
        down2 = BatchNormalization()(down2, training = train_batch_norm)
        down2 = Activation('relu')(down2)
        down2 = SeparableConv2D(128, (3, 3), padding='same')(down2)
        down2 = BatchNormalization()(down2, training = train_batch_norm)
        down2 = Activation('relu')(down2)
        down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    with K.name_scope('Down16'):
        down3 = SeparableConv2D(256, (3, 3), padding='same')(down2_pool)
        down3 = BatchNormalization()(down3, training = train_batch_norm)
        down3 = Activation('relu')(down3)
        down3 = SeparableConv2D(256, (3, 3), padding='same')(down3)
        down3 = BatchNormalization()(down3, training = train_batch_norm)
        down3 = Activation('relu')(down3)
        down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    with K.name_scope('Down8'):
        down4 = SeparableConv2D(512, (3, 3), padding='same')(down3_pool)
        down4 = BatchNormalization()(down4, training = train_batch_norm)
        down4 = Activation('relu')(down4)
        down4 = SeparableConv2D(512, (3, 3), padding='same')(down4)
        down4 = BatchNormalization()(down4, training = train_batch_norm)
        down4 = Activation('relu')(down4)
        down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    with K.name_scope('Center'):
        center = SeparableConv2D(1024, (3, 3), padding='same')(down4_pool)
        center = BatchNormalization()(center, training = train_batch_norm)
        center = Activation('relu')(center)
        center = SeparableConv2D(1024, (3, 3), padding='same')(center)
        center = BatchNormalization()(center, training = train_batch_norm)
        center = Activation('relu')(center)
    # center

    with K.name_scope('Up16'):
        up4 = UpSampling2D((2, 2))(center)
        up4 = concatenate([down4, up4], axis=3)
        up4 = Conv2D(512, (3, 3), padding='same')(up4)
        up4 = BatchNormalization()(up4, training = train_batch_norm)
        up4 = Activation('relu')(up4)
        up4 = Conv2D(512, (3, 3), padding='same')(up4)
        up4 = BatchNormalization()(up4, training = train_batch_norm)
        up4 = Activation('relu')(up4)
    # 16

    with K.name_scope('Up32'):
        up3 = UpSampling2D((2, 2))(up4)
        up3 = concatenate([down3, up3], axis=3)
        up3 = Conv2D(256, (3, 3), padding='same')(up3)
        up3 = BatchNormalization()(up3, training = train_batch_norm)
        up3 = Activation('relu')(up3)
        up3 = Conv2D(256, (3, 3), padding='same')(up3)
        up3 = BatchNormalization()(up3, training = train_batch_norm)
        up3 = Activation('relu')(up3)
    # 32

    with K.name_scope('Up64'):
        up2 = UpSampling2D((2, 2))(up3)
        up2 = concatenate([down2, up2], axis=3)
        up2 = Conv2D(128, (3, 3), padding='same')(up2)
        up2 = BatchNormalization()(up2, training = train_batch_norm)
        up2 = Activation('relu')(up2)
        up2 = Conv2D(128, (3, 3), padding='same')(up2)
        up2 = BatchNormalization()(up2, training = train_batch_norm)
        up2 = Activation('relu')(up2)
    # 64

    with K.name_scope('Up128'):
        up1 = UpSampling2D((2, 2))(up2)
        up1 = concatenate([down1, up1], axis=3)
        up1 = Conv2D(64, (3, 3), padding='same')(up1)
        up1 = BatchNormalization()(up1, training = train_batch_norm)
        up1 = Activation('relu')(up1)
        up1 = Conv2D(64, (3, 3), padding='same')(up1)
        up1 = BatchNormalization()(up1, training = train_batch_norm)
        up1 = Activation('relu')(up1)
    # 128

    with K.name_scope('Up256'):
        up0 = UpSampling2D((2, 2))(up1)
        up0 = concatenate([down0, up0], axis=3)
        up0 = Conv2D(32, (3, 3), padding='same')(up0)
        up0 = BatchNormalization()(up0, training = train_batch_norm)
        up0 = Activation('relu')(up0)
        up0 = Conv2D(32, (3, 3), padding='same')(up0)
        up0 = BatchNormalization()(up0,training = train_batch_norm)
        up0 = Activation('relu')(up0)
    # 256

    with K.name_scope('Up512'):
        up0a = UpSampling2D((2, 2))(up0)
        up0a = concatenate([down0a, up0a], axis=3)
        up0a = Conv2D(16, (3, 3), padding='same')(up0a)
        up0a = BatchNormalization()(up0a, training = train_batch_norm)
        up0a = Activation('relu')(up0a)
        up0a = Conv2D(16, (3, 3), padding='same')(up0a)
        up0a = BatchNormalization()(up0a, training = train_batch_norm)
        up0a = Activation('relu')(up0a)
    # 512

    with K.name_scope('Up1024'):
        up0b = UpSampling2D((2, 2))(up0a)
        up0b = concatenate([down0b, up0b], axis=3)
        up0b = Conv2D(8, (3, 3), padding='same')(up0b)
        up0b = BatchNormalization()(up0b, training = train_batch_norm)
        up0b = Activation('relu')(up0b)
        up0b = Conv2D(8, (3, 3), padding='same')(up0b)
        up0b = BatchNormalization()(up0b, training = train_batch_norm)
        up0b = Activation('relu')(up0b)
    # 1024

    classify = Conv2D(num_mask_channels, (1, 1), activation='sigmoid')(up0b)
    model = Model(inputs=inputs, outputs=classify)

    return model