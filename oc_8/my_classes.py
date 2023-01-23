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



# classes for data loading and preprocessing
class Dataset:
    """Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): text file containing a list of paths to images dir
        masks_dir (str): text file containing a list of paths to segmentation masks 
        
        cats_replace (dict): values of classes to replace in segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    cats_replace = {0: [0, 1, 2, 3, 4, 5, 6],
     1: [7, 8, 9, 10],
     2: [11, 12, 13, 14, 15, 16],
     3: [17, 18, 19, 20],
     4: [21, 22],
     5: [23],
     6: [24, 25],
     7: [26, 27, 28, 29, 30, 31, 32, 33, -1]
    }
    
    
    ##CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
    ##           'tree', 'signsymbol', 'fence', 'car', 
    ##           'pedestrian', 'bicyclist', 'unlabelled']
    
    DATA_PATH = './data/'
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            #classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        with open(self.DATA_PATH + images_dir, 'r') as f:
            self.images_fps = [line.strip() for line in f.readlines()]
        
        with open(self.DATA_PATH + masks_dir, 'r') as f:
            self.masks_fps = [line.strip() for line in f.readlines()]
        
        ##self.ids = os.listdir(images_dir)
        ##self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        ##self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        ## convert str names to class values on masks
        ##self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # convert mask
        mask = self._convert_mask(mask)
        
        ## extract certain classes from mask (e.g. cars)
        ##masks = [(mask == v) for v in self.class_values]
        ##mask = np.stack(masks, axis=-1).astype('float')
        ##
        ## add background if mask is not binary
        ##if mask.shape[-1] != 1:
        ##    background = 1 - mask.sum(axis=-1, keepdims=True)
        ##    mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask.astype(np.float32)
    
    def _convert_mask(self, ids_img):
        mask_labelids = pd.DataFrame(ids_img)
        for new_value, old_value in self.cats_replace.items():
            mask_labelids = mask_labelids.replace(old_value, new_value);
        mask_labelids = mask_labelids.to_numpy()
        
        clc = 8
        msk = np.zeros((mask_labelids.shape[0],mask_labelids.shape[1],clc))
        for li in np.unique(mask_labelids):
            msk[:,:,li] = np.logical_or(msk[:,:,li],(mask_labelids==li))
        return np.array(msk, dtype='uint8')
    
    def __len__(self):
        return len(self.images_fps)
    
class Dataloder(tf.keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)   


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
        
        X = np.array([cv2.resize(cv2.cvtColor(cv2.imread(path_X), cv2.COLOR_BGR2RGB), self.dim) for path_X in batch_x]).astype(np.float32)
        y = np.array([cv2.resize(self._convert_mask(cv2.imread(path_y,0)), self.dim) for path_y in batch_y]).astype(np.float32)
        
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
        return np.array(msk, dtype='uint8')
        #return np.array(mask_labelids, dtype='uint8')
        

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
            y = np.expand_dims(y, axis = 3)
        X = X /255.
        return np.array(X, dtype='uint8'), y
