if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../'))

import unittest

import numpy as np

from gantools.data.transformation import tf_flip_slices_1d, tf_flip_slices_2d, tf_flip_slices_3d, tf_patch2img_1d, tf_patch2img_2d, tf_patch2img_3d
import tensorflow as tf
tf.enable_eager_execution()


class Patch2Img(unittest.TestCase):
    def test_1d(self):
        sig1d = np.random.rand(3,4,7).astype(np.float32)
        sig1d0 = np.zeros((3,4,7)).astype(np.float32)
        # global test
        r1 = tf_patch2img_1d(sig1d, tf_flip_slices_1d(sig1d), use_symmetry=True)
        r2 = tf_patch2img_1d(sig1d, sig1d0, use_symmetry=True)
        np.testing.assert_allclose(r1,r2)
        # per sample test
        sig1d02 = np.concatenate((sig1d0[2:], tf_flip_slices_1d(sig1d)[:2]), axis=0)
        r1 = tf_patch2img_1d(sig1d, tf_flip_slices_1d(sig1d), use_symmetry=True)
        r2 = tf_patch2img_1d(sig1d, sig1d0, use_symmetry=True)
        np.testing.assert_allclose(r1,r2)

    def test_2d(self):    
        sig2dr = np.random.rand(5,4,4,3).astype(np.float32)
        sig2dl = np.random.rand(5,4,4,3).astype(np.float32)
        sig2dl2 = np.random.rand(5,4,4,3).astype(np.float32)
        sig2d0 = np.zeros((5,4,4,3)).astype(np.float32)
        # global test
        sig2dr_dl, sig2dr_ur, sig2dr_ul = tf_flip_slices_2d(sig2dr, sig2dr, sig2dr)
        r1 = tf_patch2img_2d(sig2dr, sig2dr_dl, sig2dr_ur, sig2dr_ul, use_symmetry=True)
        r2 = tf_patch2img_2d(sig2dr, sig2d0, sig2d0, sig2d0, use_symmetry=True)
        np.testing.assert_allclose(r1,r2)

        sig2dr_dl, sig2dr_ur, sig2dr_ul = tf_flip_slices_2d(sig2dr, sig2dr, sig2dr)
        _, sig2dr_dl, _ = tf_flip_slices_2d(sig2dl,sig2dl, sig2dl)
        r1 = tf_patch2img_2d(sig2dr, sig2dl, sig2dr_ur, sig2dr_dl, use_symmetry=True)
        r2 = tf_patch2img_2d(sig2dr, sig2dl, sig2d0, sig2d0, use_symmetry=True)
        np.testing.assert_allclose(r1,r2)

        sig2dr_dl, sig2dr_ur, sig2dr_ul = tf_flip_slices_2d(sig2dr, sig2dr, sig2dr)
        sig2dr_ul, _, _ = tf_flip_slices_2d(sig2dl,sig2dl, sig2dl)
        r1 = tf_patch2img_2d(sig2dr, sig2dr_dl, sig2dl, sig2dr_ul, use_symmetry=True)
        r2 = tf_patch2img_2d(sig2dr,sig2d0, sig2dl, sig2d0, use_symmetry=True)
        np.testing.assert_allclose(r1,r2)

    def test_3d(self):    
        sig3d = list(np.random.rand(5,4,4,4,3).astype(np.float32) for _ in range(8))

        sig3d0 = np.zeros((5,4,4,4,3)).astype(np.float32)
        sig3d_f = tf_flip_slices_3d(*[sig3d[0]]*7)
        args_1 = list((sig3d[0], *sig3d_f))
        args_2 = list((sig3d[0], *[sig3d0]*7))

        r1 = tf_patch2img_3d(*args_1, use_symmetry=True)
        r2 = tf_patch2img_3d(*args_2, use_symmetry=True)
        np.testing.assert_allclose(r1,r2)

if __name__ == '__main__':
    unittest.main()