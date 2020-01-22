import random
import numpy as np

class MixAug():
    '''
    different from the normal image augmentation,
    the mix image aug needs at least 2 images.
    Everytime we mix those images and their labels.
    Therefore, the images and labels here are a batch list,
    the size of batch list is based on the size of your net's batchsize
    '''

    def __init__(self):
        pass

    def img_lab_shuffle(self, images, labels):
        '''
        take a batch of images and labels as input
        output their shuffle result
        '''
        join = list(zip(images, labels))
        random.shuffle(join)
        images_shuffle, labels_shuffle = zip(*join)
        return images_shuffle, labels_shuffle

    def random_point(self, height, width, range_min=0.2, range_max=0.8):
        '''
        produce a random point in an image based on the given range
        '''
        random_height = random.randint(int(height*range_min),int(height*range_max))
        random_width = random.randint(int(width*range_min),int(width*range_max))
        
        return random_height, random_width

    def mixUp(self, images, labels):
        '''
        implement based on the mixup paper
        '''
        images_shuffle, labels_shuffle = self.img_lab_shuffle(images, labels)
        images = np.array(images)
        labels = np.array(labels)
        images_shuffle = np.array(images_shuffle)
        labels_shuffle = np.array(labels_shuffle)
        
        random_seed = 0.5
        images_mix = images*random_seed + images_shuffle*(1-random_seed)
        labels_mix = labels*random_seed + labels_shuffle*(1-random_seed)
        
        return images_mix, labels_mix
