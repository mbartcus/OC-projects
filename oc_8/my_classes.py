import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras import backend as K
#import imgaug as ia
#import imgaug.augmenters as iaa
#from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np
import cv2
import pandas as pd
#from PIL import Image
import albumentations as aug

class DataGenerator(Sequence):

    cats_replace = {0: [0, 1, 2, 3, 4, 5, 6],
     1: [7, 8, 9, 10],
     2: [11, 12, 13, 14, 15, 16],
     3: [17, 18, 19, 20],
     4: [21, 22],
     5: [23],
     6: [24, 25],
     7: [26, 27, 28, 29, 30, 31, 32, 33, -1]
    }

    def __init__(self, images_path, labels_path, batch_size, dim=(256,128), shuffle=True, augmentation=False):
        self.images_path = images_path  # liste contenant les chemins des images
        self.labels_path = labels_path # liste contenant les chemins des masques
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images_path))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        # Counts the number of possible batches that can be made from the total available datasets in list_IDs
        # Rule of thumb, num_datasets % batch_size = 0, so every sample is seen
        return int(np.floor(len(self.images_path) / self.batch_size))


    def __getitem__(self, index):

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_x = [self.images_path[k] for k in indexes]
        batch_y = [self.labels_path[k] for k in indexes]

        X, y = self.__data_generation(batch_x, batch_y)
        

        return (X, y)

    def __data_generation(self, batch_x, batch_y):
        
        #X = np.array([np.array(Image.open(image_name).resize(self.dim, Image.Resampling.LANCZOS)) for image_name in batch_x])
                
        #y = np.array([np.array(Image.fromarray(self._convert_mask(np.array(Image.open(mask_name))).astype('uint8'), 'RGB').resize(self.dim))     for mask_name in batch_y])
        
        X = np.array([cv2.resize(cv2.cvtColor(cv2.imread(path_X), cv2.COLOR_BGR2RGB), self.dim) for path_X in batch_x])
        y = np.array([cv2.resize(self._convert_mask(cv2.imread(path_y,0)), self.dim) for path_y in batch_y])
        
        print(y.shape)
        
        if self.augmentation:
            X, y = self._augment_data(X,y)

        return self._transform_data(X,y)

    def _convert_mask(self, ids_img):
        mask_labelids = pd.DataFrame(ids_img)
        for new_value, old_value in self.cats_replace.items():
            mask_labelids = mask_labelids.replace(old_value, new_value);
        mask_labelids = mask_labelids.to_numpy()
        
        clc = 8
        msk = np.zeros((mask_labelids.shape[0],mask_labelids.shape[1],clc))
        for li in np.unique(mask_labelids):
            msk[:,:,li] = np.logical_or(msk[:,:,li],(mask_labelids==li))
        #print(np.array(msk, dtype='uint8').shape)
        return np.array(msk, dtype='uint8')
        

    def _augment_data(self, X, y):
        seq = iaa.Sequential([
            iaa.Sometimes( # Symétrie verticale sur 50% des images
                0.5,
                iaa.Fliplr(0.5)
            ),
            iaa.Sometimes( # Flou gaussien sur 50% des images
                0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            iaa.LinearContrast((0.75, 1.5)), # Modifie le contraste
            iaa.AdditiveGaussianNoise(scale=(0.0, 0.2*255)), # Ajout d'un bruit gaussien
            iaa.Multiply((0.8, 1.2)), # Rend l'image plus sombre ou plus claire
            iaa.Affine( # Zoom, translation, rotation
                scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-15, 15)
            )
        ], random_order=True) # apply augmenters in random order

        new_X = []
        new_y = []

        for i in range(len(X)):
            img = X[i]
            mask = y[i]
            new_X.append(img)
            new_y.append(mask)
            segmap = SegmentationMapsOnImage(mask, shape=img.shape)

            imag_aug_i, segmap_aug_i = seq(image=img, segmentation_maps=segmap)
            new_X.append(imag_aug_i)
            new_y.append(segmap_aug_i.get_arr())

        new_X = np.array(new_X)
        new_y = np.array(new_y)

        return new_X, new_y

    def _transform_data(self,X,y):
        if len(y.shape) == 3:
            print('1')
            y = np.expand_dims(y, axis = 3)
            print(y.shape)
        X = X /255.
        print('fin')
        return np.array(X, dtype='uint8'), y
