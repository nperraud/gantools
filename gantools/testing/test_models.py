if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import unittest

from gantools.model import WGAN, LapWGAN, UpscalePatchWGAN
from gantools.gansystem import GANsystem
from gantools.data.core import Dataset
import tensorflow as tf
import numpy as np


class TestGANmodels(unittest.TestCase):
    def test_default_params_wgan(self):
        obj = GANsystem(WGAN)

    def test_default_params_lapgan(self):
        obj = GANsystem(LapWGAN)

    def test_default_params_patchgan(self):
        obj = GANsystem(UpscalePatchWGAN)

    def test_1d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [16, 1]  # Shape of the image
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 100
        params['net']['generator']['full'] = [2 * 8]
        params['net']['generator']['nfilter'] = [2, 32, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn, bn]
        params['net']['generator']['shape'] = [[5], [5], [5], [5]]
        params['net']['generator']['stride'] = [1, 2, 1, 1]
        params['net']['generator']['data_size'] = 1
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32, 32, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn, bn, bn]
        params['net']['discriminator']['shape'] = [[5], [5], [5], [3]]
        params['net']['discriminator']['stride'] = [2, 2, 2, 1]
        params['net']['discriminator']['data_size'] = 1

        X = np.random.rand(101, 16)
        dataset = Dataset(X)
        wgan = GANsystem(WGAN, params)
        wgan.train(dataset)
        img = wgan.generate(2)
        assert (len(img) == 2)
        assert (img.shape[1:] == (16, 1))
        img = wgan.generate(500)
        assert (len(img) == 500)

    def test_2d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [16, 16, 1]  # Shape of the image
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 100
        params['net']['generator']['full'] = [2 * 8 * 8]
        params['net']['generator']['nfilter'] = [2, 32, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn, bn]
        params['net']['generator']['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5]]
        params['net']['generator']['stride'] = [1, 2, 1, 1]
        params['net']['generator']['data_size'] = 2
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32, 32, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn, bn, bn]
        params['net']['discriminator']['shape'] = [[5, 5], [5, 5], [5, 5],
                                                   [3, 3]]
        params['net']['discriminator']['stride'] = [2, 2, 2, 1]
        params['net']['discriminator']['data_size'] = 2

        X = np.random.rand(101, 16, 16)
        dataset = Dataset(X)
        wgan = GANsystem(WGAN, params)
        wgan.train(dataset)
        img = wgan.generate(2)
        assert (len(img) == 2)
        assert (img.shape[1:] == (16, 16, 1))
        img = wgan.generate(500)
        assert (len(img) == 500)

    def test_hinge(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [16, 16, 1]  # Shape of the image
        params['net']['loss_type'] = 'hinge'  # Shape of the image
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 100
        params['net']['generator']['full'] = [2 * 8 * 8]
        params['net']['generator']['nfilter'] = [2, 32, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn, bn]
        params['net']['generator']['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5]]
        params['net']['generator']['stride'] = [1, 2, 1, 1]
        params['net']['generator']['data_size'] = 2
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32, 32, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn, bn, bn]
        params['net']['discriminator']['shape'] = [[5, 5], [5, 5], [5, 5],
                                                   [3, 3]]
        params['net']['discriminator']['stride'] = [2, 2, 2, 1]
        params['net']['discriminator']['data_size'] = 2

        X = np.random.rand(101, 16, 16)
        dataset = Dataset(X)
        wgan = GANsystem(WGAN, params)
        wgan.train(dataset)
        img = wgan.generate(2)
        assert (len(img) == 2)
        assert (img.shape[1:] == (16, 16, 1))
        img = wgan.generate(500)
        assert (len(img) == 500)


    def test_3d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [8, 8, 8, 1]  # Shape of the image
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 100
        params['net']['generator']['full'] = [2 * 4 * 4 * 4]
        params['net']['generator']['nfilter'] = [2, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[3, 3, 3], [3, 3, 3],
                                               [5, 5, 5]]
        params['net']['generator']['stride'] = [1, 2, 1]
        params['net']['generator']['data_size'] = 3
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn, bn]
        params['net']['discriminator']['shape'] = [[5, 5, 5], [3, 3, 3],
                                                   [3, 3, 3]]
        params['net']['discriminator']['stride'] = [2, 2, 2]
        params['net']['discriminator']['data_size'] = 3

        X = np.random.rand(101, 8, 8, 8)
        dataset = Dataset(X)
        wgan = GANsystem(WGAN, params)
        wgan.train(dataset)
        img = wgan.generate(2)
        assert (len(img) == 2)
        assert (img.shape[1:] == (8, 8, 8, 1))
        img = wgan.generate(500)
        assert (len(img) == 500)


    def test_lapgan1d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [16, 1]
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 16 * 16
        params['net']['generator']['full'] = []
        params['net']['generator']['nfilter'] = [8, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[5], [5], [5]]
        params['net']['generator']['stride'] = [1, 1, 1]
        params['net']['generator']['data_size'] = 1
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn]
        params['net']['discriminator']['shape'] = [[5], [3]]
        params['net']['discriminator']['stride'] = [2, 2]
        params['net']['discriminator']['data_size'] = 1
        params['net']['upscaling'] = 2

        X = np.random.rand(101, 16)
        dataset = Dataset(X)
        wgan = GANsystem(LapWGAN, params)
        wgan.train(dataset)
        X_down = np.random.rand(500, 8, 1)
        img = wgan.generate(N=2, X_down=X_down[:2])
        assert (len(img) == 2)
        assert (img.shape[1:] == (16, 1))
        img = wgan.generate(N=500, X_down=X_down[:500])
        assert (len(img) == 500)

    def test_lapgan2d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [16, 16, 1]
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 16 * 16
        params['net']['generator']['full'] = []
        params['net']['generator']['nfilter'] = [8, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[5, 5], [5, 5], [5, 5]]
        params['net']['generator']['stride'] = [1, 1, 1]
        params['net']['generator']['data_size'] = 2
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn]
        params['net']['discriminator']['shape'] = [[5, 5], [3, 3]]
        params['net']['discriminator']['stride'] = [2, 2]
        params['net']['discriminator']['data_size'] = 2
        params['net']['upscaling'] = 2

        X = np.random.rand(101, 16, 16)
        dataset = Dataset(X)

        wgan = GANsystem(LapWGAN, params)
        wgan.train(dataset)
        X_down = np.random.rand(500, 8, 8, 1)
        img = wgan.generate(N=2, X_down=X_down[:2])
        assert (len(img) == 2)
        assert (img.shape[1:] == (16, 16, 1))
        img = wgan.generate(N=500, X_down=X_down[:500])
        assert (len(img) == 500)

    def test_lapgan3d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [8, 8, 8, 1]
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 8 * 8 * 8
        params['net']['generator']['full'] = []
        params['net']['generator']['nfilter'] = [8, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[3, 3, 3], [3, 3, 3],
                                               [3, 3, 3]]
        params['net']['generator']['stride'] = [1, 1, 1]
        params['net']['generator']['data_size'] = 3
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn]
        params['net']['discriminator']['shape'] = [[3, 3, 3], [3, 3, 3]]
        params['net']['discriminator']['stride'] = [2, 2]
        params['net']['discriminator']['data_size'] = 3
        params['net']['upscaling'] = 2

        X = np.random.rand(101, 8, 8, 8)
        dataset = Dataset(X)

        wgan = GANsystem(LapWGAN, params)
        wgan.train(dataset)
        X_down = np.random.rand(500, 4, 4, 4, 1)
        img = wgan.generate(N=2, X_down=X_down[:2])
        assert (len(img) == 2)
        assert (img.shape[1:] == (8, 8, 8, 1))
        img = wgan.generate(N=500, X_down=X_down[:500])
        assert (len(img) == 500)

    def test_patchgan1d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [8, 2]
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 8
        params['net']['generator']['full'] = []
        params['net']['generator']['nfilter'] = [8, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[3], [3], [3]]
        params['net']['generator']['stride'] = [1, 1, 1]
        params['net']['generator']['data_size'] = 1
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn]
        params['net']['discriminator']['shape'] = [[5], [3]]
        params['net']['discriminator']['stride'] = [2, 2]
        params['net']['discriminator']['data_size'] = 1
        params['net']['upscaling'] = None

        X = np.random.rand(101, 8, 2)
        dataset = Dataset(X)
        wgan = GANsystem(UpscalePatchWGAN, params)
        wgan.train(dataset)
        borders = np.random.rand(500, 8, 1)
        img = wgan.generate(N=2, borders=borders[:2])
        assert (len(img) == 2)
        assert (img.shape[1:] == (8, 1))
        img = wgan.generate(N=500, borders=borders[:500])
        assert (len(img) == 500)

    def test_patchgan2d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [8, 8, 4]
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 8 * 8
        params['net']['generator']['full'] = []
        params['net']['generator']['nfilter'] = [8, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[3, 3], [3, 3], [3, 3]]
        params['net']['generator']['stride'] = [1, 1, 1]
        params['net']['generator']['data_size'] = 2
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn]
        params['net']['discriminator']['shape'] = [[5, 5], [3, 3]]
        params['net']['discriminator']['stride'] = [2, 2]
        params['net']['discriminator']['data_size'] = 2
        params['net']['upscaling'] = None

        X = np.random.rand(101, 8, 8, 4)
        dataset = Dataset(X)

        wgan = GANsystem(UpscalePatchWGAN, params)
        wgan.train(dataset)
        borders = np.random.rand(500, 8, 8, 3)
        img = wgan.generate(N=2, borders=borders[:2])
        assert (len(img) == 2)
        assert (img.shape[1:] == (8, 8, 1))
        img = wgan.generate(N=500, borders=borders[:500])
        assert (len(img) == 500)

    def test_patchgan3d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [8, 8, 8, 8]
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 8 * 8 * 8
        params['net']['generator']['full'] = []
        params['net']['generator']['nfilter'] = [8, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[3, 3, 3], [3, 3, 3],
                                               [3, 3, 3]]
        params['net']['generator']['stride'] = [1, 1, 1]
        params['net']['generator']['data_size'] = 3
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn]
        params['net']['discriminator']['shape'] = [[3, 3, 3], [3, 3, 3]]
        params['net']['discriminator']['stride'] = [2, 2]
        params['net']['discriminator']['data_size'] = 3
        params['net']['upscaling'] = None

        X = np.random.rand(101, 8, 8, 8, 8)
        dataset = Dataset(X)

        wgan = GANsystem(UpscalePatchWGAN, params)
        wgan.train(dataset)
        borders = np.random.rand(500, 8, 8, 8, 7)
        img = wgan.generate(N=2, borders=borders[:2])
        assert (len(img) == 2)
        assert (img.shape[1:] == (8, 8, 8, 1))
        img = wgan.generate(N=500, borders=borders[:500])
        assert (len(img) == 500)

    def test_patchupscalegan1d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [8, 2]
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 8
        params['net']['generator']['full'] = []
        params['net']['generator']['nfilter'] = [8, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[3], [3], [3]]
        params['net']['generator']['stride'] = [1, 1, 1]
        params['net']['generator']['data_size'] = 1
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn]
        params['net']['discriminator']['shape'] = [[5], [3]]
        params['net']['discriminator']['stride'] = [2, 2]
        params['net']['discriminator']['data_size'] = 1
        params['net']['upscaling'] = 2

        X = np.random.rand(101, 8, 2)
        dataset = Dataset(X)
        wgan = GANsystem(UpscalePatchWGAN, params)
        wgan.train(dataset)
        borders = np.random.rand(500, 8, 1)
        X_down = np.random.rand(500, 4, 1)
        img = wgan.generate(N=2, X_down=X_down[:2], borders=borders[:2])
        assert (len(img) == 2)
        assert (img.shape[1:] == (8, 1))
        img = wgan.generate(N=500, X_down=X_down[:500], borders=borders[:500])
        assert (len(img) == 500)

    def test_patchupscalegan2d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [8, 8, 4]
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 8 * 8
        params['net']['generator']['full'] = []
        params['net']['generator']['nfilter'] = [8, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[3, 3], [3, 3], [3, 3]]
        params['net']['generator']['stride'] = [1, 1, 1]
        params['net']['generator']['data_size'] = 2
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn]
        params['net']['discriminator']['shape'] = [[5, 5], [3, 3]]
        params['net']['discriminator']['stride'] = [2, 2]
        params['net']['discriminator']['data_size'] = 2
        params['net']['upscaling'] = 2

        X = np.random.rand(101, 8, 8, 4)
        dataset = Dataset(X)

        wgan = GANsystem(UpscalePatchWGAN, params)
        wgan.train(dataset)
        borders = np.random.rand(500, 8, 8, 3)
        X_down = np.random.rand(500, 4, 4, 1)
        img = wgan.generate(N=2, X_down=X_down[:2], borders=borders[:2])
        assert (len(img) == 2)
        assert (img.shape[1:] == (8, 8, 1))
        img = wgan.generate(N=500, X_down=X_down[:500], borders=borders[:500])
        assert (len(img) == 500)

    def test_patchupscalegan3d(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [8, 8, 8, 8]
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 8 * 8 * 8
        params['net']['generator']['full'] = []
        params['net']['generator']['nfilter'] = [8, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[3, 3, 3], [3, 3, 3],
                                               [3, 3, 3]]
        params['net']['generator']['stride'] = [1, 1, 1]
        params['net']['generator']['data_size'] = 3
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn]
        params['net']['discriminator']['shape'] = [[3, 3, 3], [3, 3, 3]]
        params['net']['discriminator']['stride'] = [2, 2]
        params['net']['discriminator']['data_size'] = 3
        params['net']['upscaling'] = 2

        X = np.random.rand(101, 8, 8, 8, 8)
        dataset = Dataset(X)


        wgan = GANsystem(UpscalePatchWGAN, params)
        wgan.train(dataset)
        borders = np.random.rand(500, 8, 8, 8, 7)
        X_down = np.random.rand(500, 4, 4, 4, 1)

        img = wgan.generate(N=2, X_down=X_down[:2], borders=borders[:2])
        assert (len(img) == 2)
        assert (img.shape[1:] == (8, 8, 8, 1))
        img = wgan.generate(N=500, X_down=X_down[:500], borders=borders[:500])
        assert (len(img) == 500)


    def test_patchupscalegan1dborder(self):
        bn = False
        params = dict()
        params['optimization'] = dict()
        params['optimization']['epoch'] = 1
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['net'] = dict()
        params['net']['shape'] = [8, 2]
        params['net']['generator'] = dict()
        params['net']['generator']['latent_dim'] = 8
        params['net']['generator']['full'] = [16]
        params['net']['generator']['nfilter'] = [8, 32, 1]
        params['net']['generator']['batch_norm'] = [bn, bn]
        params['net']['generator']['shape'] = [[3], [3], [3]]
        params['net']['generator']['stride'] = [1, 1, 1]
        params['net']['generator']['data_size'] = 1
        params['net']['generator']['borders'] = dict()
        params['net']['generator']['borders']['width_full'] = None
        params['net']['generator']['borders']['nfilter'] = [4, 1]
        params['net']['generator']['borders']['batch_norm'] = [bn, bn]
        params['net']['generator']['borders']['shape'] = [[5], [3]]
        params['net']['generator']['borders']['stride'] = [2, 2]
        params['net']['generator']['borders']['data_size'] = 1
        params['net']['generator']['borders']['width_full'] = 2
        params['net']['discriminator'] = dict()
        params['net']['discriminator']['full'] = [32]
        params['net']['discriminator']['nfilter'] = [16, 32]
        params['net']['discriminator']['batch_norm'] = [bn, bn]
        params['net']['discriminator']['shape'] = [[5], [3]]
        params['net']['discriminator']['stride'] = [2, 2]
        params['net']['discriminator']['data_size'] = 1
        params['net']['upscaling'] = 2


        X = np.random.rand(101, 8, 2)
        dataset = Dataset(X)
        wgan = GANsystem(UpscalePatchWGAN, params)
        wgan.train(dataset)
        borders = np.random.rand(500, 8, 1)
        X_smooth = np.random.rand(500, 8, 1)
        print(borders[:2].shape)
        img = wgan.generate(N=2, X_smooth=X_smooth[:2], borders=borders[:2])
        assert (len(img) == 2)
        assert (img.shape[1:] == (8, 1))
        img = wgan.generate(N=500, X_smooth=X_smooth[:500], borders=borders[:500])
        assert (len(img) == 500)

    def test_conditionalwgan(self):

        bn = False
        ns = 16

        # Parameters for the generator
        params_generator = dict()
        params_generator['latent_dim'] = 126
        params_generator['stride'] = [1, 2, 1, 1]
        params_generator['nfilter'] = [32, 64, 32, 1]
        params_generator['shape'] = [[5, 5], [5, 5], [5, 5], [5, 5]]
        params_generator['batch_norm'] = [bn, bn, bn]
        params_generator['full'] = [8 * 8 * 32]
        params_generator['non_lin'] = tf.nn.relu
        params_generator['in_conv_shape'] = [8, 8]

        # Parameters for the discriminator
        params_discriminator = dict()
        params_discriminator['stride'] = [2, 2, 1]
        params_discriminator['nfilter'] = [16, 64, 32]
        params_discriminator['shape'] = [[5, 5], [5, 5], [3, 3]]
        params_discriminator['batch_norm'] = [bn, bn, bn]
        params_discriminator['full'] = [512, 128, 32]
        params_discriminator['minibatch_reg'] = False

        # Optimization parameters
        d_opt = dict()
        d_opt['optimizer'] = "rmsprop"
        d_opt['learning_rate'] = 3e-5
        params_optimization = dict()
        params_optimization['n_critic'] = 5
        params_optimization['batch_size'] = 8
        params_optimization['epoch'] = 1

        # Cosmology parameters
        params_cosmology = dict()
        params_cosmology['forward_map'] = None
        params_cosmology['backward_map'] = None

        # all parameters
        params = dict()
        params['net'] = dict() # All the parameters for the model
        params['net']['generator'] = params_generator
        params['net']['discriminator'] = params_discriminator
        params['net']['cosmology'] = params_cosmology # Parameters for the cosmological summaries
        params['net']['shape'] = [ns, ns, 1] # Shape of the image
        params['net']['gamma_gp'] = 10 # Gradient penalty

        # Conditional params
        params['net']['prior_normalization'] = False
        params['net']['cond_params'] = 2
        params['net']['init_range'] = [[0, 1], [0, 1]]
        params['net']['prior_distribution'] = "gaussian_length"
        params['net']['final_range'] = [0.1*np.sqrt(params_generator['latent_dim']), 1*np.sqrt(params_generator['latent_dim'])]

        params['optimization'] = params_optimization
        params['summary_every'] = 4
        params['save_every'] = 5
        params['print_every'] = 3
        params['Nstats'] = 100

        X = np.random.rand(101, ns, ns)
        parameters = np.random.rand(101, 2)
        dataset = Dataset_parameters(X, parameters)
        wgan = GANsystem(ConditionalParamWGAN, params)
        wgan.train(dataset)
        img = wgan.generate(N=16, **{'z': wgan.net.sample_latent(bs=16, params=np.random.rand(16, 2))})
        assert (img.shape[0] == 16 and img.shape[1] == ns and img.shape[2] == ns and img.shape[3] == 1)

if __name__ == '__main__':
    unittest.main()