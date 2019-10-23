
if __name__ == '__main__':
    import sys, os
    import shutil
    import tensorflow as tf
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../'))


import unittest

from gantools.testing import test_downsample
from gantools.testing import test_split
from gantools.testing import test_slices
from gantools.testing import test_gansystem
from gantools.testing import test_models
from gantools.testing import test_utils
from gantools.testing import test_plots
from gantools.testing import test_metric

loader = unittest.TestLoader()

suites = []
suites.append(loader.loadTestsFromModule(test_downsample))
suites.append(loader.loadTestsFromModule(test_split))
suites.append(loader.loadTestsFromModule(test_slices))
suites.append(loader.loadTestsFromModule(test_gansystem))
suites.append(loader.loadTestsFromModule(test_utils))
suites.append(loader.loadTestsFromModule(test_models))
suites.append(loader.loadTestsFromModule(test_plots))
suites.append(loader.loadTestsFromModule(test_metric))
suite = unittest.TestSuite(suites)


def run():  # pragma: no cover
    unittest.TextTestRunner(verbosity=2).run(suite)

def saferm(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        
def clean():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    saferm(os.path.join(dir_path,'checkpoints'))
    saferm(os.path.join(dir_path,'__pycache__'))

if __name__ == '__main__':  # pragma: no cover
    os.environ["CUDA_VISIBLE_DEVICES"]=""

    run()
    clean()
    
    