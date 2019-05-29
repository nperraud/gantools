import itertools
import numpy as np
from . import transformation
import functools
from gantools.utils import compose2


def do_nothing(x):
    return x



class DatasetMedical(object):
    ''' Dataset object for GAN and CosmoGAN classes

        Transform should probably be False for a classical GAN.
    '''

    def __init__(self, X, spix=32, shuffle=True, augmentation=True, transform=None, dtype=np.uint8):
        ''' Initialize a Dataset object

        Arguments
        ---------
        * X         : numpy array containing the data
        * shuffle   : True if the data should be shuffled
        * augmentation : Function to be applied ot each batch of the dataset
                      This allows extend the dataset.
        * slice_fn : Slicing function to cut the data into smaller parts
        '''
        X = X.astype(dtype)
        self._shuffle = shuffle
        # if slice_fn:
        # else:
        #     self._slice_fn = do_noting
        slice_fn = functools.partial(slice_shift_3d_patch, spix=spix)
        if augmentation:
            self._slice_fn = compose2(transformation.random_rotate_3d, slice_fn)
        else:
            self._slice_fn = slice_fn

        if transform:
            self._transform = transform
        else:
            self._transform = do_nothing
        self._spix = spix

        self._N = len(self._slice_fn(X))
        if shuffle:
            self._p = np.random.permutation(self._N)
        else:
            self._p = np.arange(self._N)
        self._X = X

    def get_all_data(self):
        ''' Return all the data (shuffled) '''
        return self._transform(self._slice_fn(self._X, self.spix)[self._p])

    def get_samples(self, N=100):
        ''' Get the `N` first samples '''
        return self._transform(self._slice_fn(self._X)[self._p[:N]])

    # TODO: kwargs to be removed
    def iter(self, batch_size=1, **kwargs):
        return self.__iter__(batch_size, **kwargs)

    # TODO: kwargs to be removed
    def __iter__(self, batch_size=1, **kwargs):

        if batch_size > self.N:
            raise ValueError(
                'Batch size greater than total number of samples available!')

        # Reshuffle the data
        if self.shuffle:
            self._p = np.random.permutation(self._N)
        nel = (self._N // batch_size) * batch_size
        transformed_data = self._slice_fn(self._X)[self._p[range(nel)]]
        for data in grouper(transformed_data, batch_size):
            yield self._transform(np.array(data))

    @property
    def shuffle(self):
        ''' Is the dataset suffled? '''
        return self._shuffle

    @property
    def N(self):
        ''' Number of element in the dataset '''
        return self._N

    @property
    def spix(self):
        return self._spix
    

def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks. This function commes
    from itertools
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)


def slice_shift_3d_patch(bbox, spix=32, limit_size=True):
    '''
    cubes: the 3d histograms - [:, :, :, :]
    '''

    # Handle the dimesnsions
    l = len(bbox.shape)
    if l < 3:
        ValueError('Not enough dimensions')
    elif l == 3:
        bbox = bbox.reshape([1, *bbox.shape]) # add one extra dimension for number of cubes
    elif l > 4:
        ValueError('To many dimensions')

    _, sx, sy, sz = bbox.shape
    nx = (sx // spix) - 1
    ny = (sy // spix) - 1
    nz = (sz // spix) - 1
    if limit_size:
        if nx>10:
            nx=10    
        if ny>10:
            ny=10
        if nz>10:
            nz=10
    lx = sx - nx*spix
    ly = sy - ny*spix
    lz = sz - nz*spix
    
    # 0) Select a subpart of the images
    rx = np.random.randint(0, lx)
    ry = np.random.randint(0, ly)
    rz = np.random.randint(0, lz)
    bbox = bbox[:, rx:rx+nx*spix, ry:ry+ny*spix, rz:rz+nz*spix]

    # 1) Create all 7 neighbors for each smaller cube
    img1 = np.roll(bbox, spix, axis=2)
    img1[:, :, :spix, :] = 0

    img2 = np.roll(bbox, spix, axis=3)
    img2[:, :, :, :spix] = 0
    
    img3 = np.roll(img1, spix, axis=3)
    img3[:, :, :, :spix] = 0
    
    img4 = np.roll(bbox, spix, axis=1) # extra for the 3D case
    img4[:, :spix, :, :] = 0
    
    img5 = np.roll(img4, spix, axis=2)
    img5[:, :, :spix, :] = 0
    
    img6 = np.roll(img4, spix, axis=3)
    img6[:, :, :, :spix] = 0
    
    img7 = np.roll(img5, spix, axis=3)
    img7[:, :, :, :spix] = 0

    # 2) Concatenate
    img_with_nbrs = np.stack([bbox, img1, img2, img3, img4, img5, img6, img7], axis=4) # 7 neighbors plus the original cube

    # Clear variable to gain some RAM
    del img1, img2, img3, img4, img5, img6, img7
    
    # 3) Slice the cubes
    img_with_nbrs = np.vstack(np.split(img_with_nbrs, nx, axis=1))
    img_with_nbrs = np.vstack(np.split(img_with_nbrs, ny, axis=2))
    img_with_nbrs = np.vstack(np.split(img_with_nbrs, nz, axis=3))
      
    return img_with_nbrs

def slice_3d(cubes, spix=64):
    '''
    slice each cube in cubes to smaller cubes,
    and return all the smaller cubes
    '''


    _, sx, sy, sz = cubes.shape
    nx = (sx // spix) - 1
    ny = (sy // spix) - 1
    nz = (sz // spix) - 1
    lx = sx - nx*spix
    ly = sy - ny*spix
    lz = sz - nz*spix
    
    # 0) Select a subpart of the images
    rx = np.random.randint(0, lx)
    ry = np.random.randint(0, ly)
    rz = np.random.randint(0, lz)
    cubes = cubes[:, rx:rx+nx*spix, ry:ry+ny*spix, rz:rz+nz*spix]

    # split along first dimension
    cubes = np.vstack(np.split(cubes, nx, axis=1))
    # split along second dimension
    cubes = np.vstack(np.split(cubes, ny, axis=2))
    # split along third dimension
    cubes = np.vstack(np.split(cubes, nz, axis=3))

    return cubes

