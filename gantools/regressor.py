import tensorflow as tf
import numpy as np
# The next import should be changed
from tfnntools.model import BaseNet, rprint
from tfnntools.nnsystem import NNSystem

from gantools.blocks import *
from gantools.data import Dataset_parameters
from gantools.utils import load_dict_pickle
from gantools.model import get_conv

from copy import deepcopy

class BaseRegressor(BaseNet):
    """Abstract class for the model."""
    def __init__(self, params={}, name='BaseRegressor'):
        self._summary = None
        self._loss = None
        super().__init__(params=params, name=name)

    @property
    def summary(self):
        return self._summary


class Regressor(BaseRegressor):

    def default_params(self):
        d_params = deepcopy(super().default_params())
        d_params['shape'] = [16, 16, 1] # Shape of the image
        d_params['cond_params'] = 1 # Number of outputs

        bn = False

        d_params['regressor'] = dict()
        d_params['regressor']['full'] = [32]
        d_params['regressor']['nfilter'] = [16, 32, 32, 32]
        d_params['regressor']['batch_norm'] = [bn, bn, bn, bn]
        d_params['regressor']['shape'] = [[5, 5], [5, 5], [5, 5], [3, 3]]
        d_params['regressor']['stride'] = [2, 2, 2, 1]
        d_params['regressor']['summary'] = True
        d_params['regressor']['data_size'] = 2 # 1 for 1D signal, 2 for images, 3 for 3D
        d_params['regressor']['activation'] = lrelu # leaky relu
        d_params['regressor']['non_lin'] = None # non linearity at the beginning of the regressor
        d_params['regressor']['minibatch_reg'] = False # Use minibatch regularization
        d_params['regressor']['spectral_norm'] = False # use spectral norm
        d_params['regressor']['weights'] = None
        d_params['regressor']['loss'] = tf.losses.mean_squared_error

        return d_params

    def __init__(self, params, name='Regressor'):
        super().__init__(params=params, name=name)

    def _build_net(self):
        self._data_size = self.params['regressor']['data_size']
        
        # Define input
        shape = self._params['shape']
        self.X = tf.placeholder(tf.float32, shape=[None, *shape], name='X')
        self.X_param = tf.placeholder(tf.float32, shape=[None, self._params['cond_params']], name='Xparam')

        # Build net
        self.R_param = self.regressor(self.X, reuse=False)

        self._loss = self._build_loss()
        self._inputs = (self.X)
        self._outputs = (self.R_param, self._R_features)

    def _build_loss(self):
        if self.params['regressor']['weights'] == 'relative':
            return self.params['regressor']['loss'](self.X_param / self.X_param, self.R_param / self.X_param)
        elif self.params['regressor']['weights'] is not None:
            return self.params['regressor']['loss'](self.X_param, self.R_param, weights=tf.constant(self.params['regressor']['weights']))
        return self.params['regressor']['loss'](self.X_param, self.R_param)

    def _add_summary(self):
        tf.summary.scalar("Regr/Loss", self._loss, collections=["train"])
        tf.summary.scalar("Regr/L2", tf.losses.mean_squared_error(self.R_param[:, :2], self.X_param), collections=["train"])

    def regressor(self, X, **kwargs):
        return regressor(X, self.params['regressor'], self.params['cond_params'], model=self, **kwargs) 

    def assert_image(self, x):
        dim = self.data_size + 1
        if len(x.shape) < dim:
            raise ValueError('The size of the data is wrong')
        elif len(x.shape) < (dim +1):
            x = np.expand_dims(x, dim)
        return x

    def batch2dict(self, batch):
        d = dict()
        d['X'] = self.assert_image(np.array([batch[i][0] for i in range(len(batch))]))
        d['X_param'] = np.array([batch[i][1][:self.params['cond_params']] for i in range(len(batch))])
        return d

    @property
    def data_size(self):
        return self._data_size


def regressor(x, params, out_dim, reuse=True, scope="regressor", model=None):
    conv = get_conv(params['data_size'])

    assert(len(params['stride']) ==
           len(params['nfilter']) ==
           len(params['batch_norm']))
    nconv = len(params['stride'])
    nfull = len(params['full'])

    with tf.variable_scope(scope, reuse=reuse):
        rprint('Regressor \n'+''.join(['-']*50), reuse)
        rprint('     The input is of size {}'.format(x.shape), reuse)
        
        if params['non_lin']:
            non_lin_f = getattr(tf, params['non_lin'])
            x = non_lin_f(x)
            rprint('    Non lienarity: {}'.format(params['non_lin']), reuse)

        for i in range(nconv):
            # TODO: this really needs to be cleaned uy...
            x = conv(x,
                     nf_out=params['nfilter'][i],
                     shape=params['shape'][i],
                     stride=params['stride'][i],
                     use_spectral_norm=params['spectral_norm'],
                     name='{}_conv'.format(i),
                     summary=params['summary'])
            rprint('     {} Conv layer with {} channels'.format(i, params['nfilter'][i]), reuse)

            if params['batch_norm'][i]:
                x = batch_norm(x, name='{}_bn'.format(i), train=True)
                rprint('         Batch norm', reuse)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)

            x = params['activation'](x)

            if model is not None:
                setattr(model, '_R_conv_activation_' + str(i), x)

        x = reshape2d(x, name='img2vec')
        rprint('     Reshape to {}'.format(x.shape), reuse)

        for i in range(nfull):
            x = linear(x,
                       params['full'][i],
                       '{}_full'.format(i+nconv),
                       summary=params['summary'])
            x = params['activation'](x)
            rprint('     {} Full layer with {} outputs'.format(nconv+i, params['full'][i]), reuse)
            rprint('         Size of the variables: {}'.format(x.shape), reuse)

            if model is not None:
                setattr(model, '_R_full_activation_' + str(i), x)

        if params['minibatch_reg']:
            rprint('     Minibatch regularization', reuse)
            x = mini_batch_reg(x, n_kernels=150, dim_per_kernel=30)
            rprint('       Size of the variables: {}'.format(x.shape), reuse)

        if model is not None:
            setattr(model, '_R_features', x)

        rprint('     {} Full layer with {} outputs'.format(nconv+nfull, 1), reuse)
        x = linear(x, out_dim, 'out', summary=params['summary'])
        rprint('     The output is of size {}'.format(x.shape), reuse)
        rprint(''.join(['-']*50)+'\n', reuse)

        return x


# Get regressor outputs and features
def get_regressor_outputs(model, images, params=None, checkpoint=None, batch_size=None):
    # Trick feed random parameters if they are not specified
    if params is None:
        params = np.random.rand((len(images), model.net.params['cond_params']))
    dat = Dataset_parameters(images, params, shuffle=False)
    if batch_size is None or batch_size > len(images):
        batch_size = len(images)
    it = dat.iter(batch_size)
    out1 = []
    out2 = []
    for i in range(len(images) // batch_size):
        batch = next(it)
        o1, o2 = model.outputs(checkpoint=checkpoint, **model.net.batch2dict(batch))
        out1.append(o1)
        out2.append(o2)
    out1 = np.vstack(out1)
    out2 = np.vstack(out2)
    return out1, out2


# Load regressor
def load_regressor(regressor_path, reg_class="Regressor"):
    if reg_class == "Regressor":
        reg_class = Regressor
    else:
        raise ValueError("Unknown class " + reg_class)
    reg_params = load_dict_pickle(regressor_path + 'params.pkl')
    reg_params['save_dir'] = regressor_path
    return NNSystem(reg_class, reg_params)