"""Main GAN module."""

import tensorflow as tf
import numpy as np
import time
import itertools
from copy import deepcopy
import yaml
from gantools.metric import ganlist
from gantools import blocks
from tfnntools.nnsystem import NNSystem

class GANsystem(NNSystem):
    """General GAN System class.

    This class contains all the method to train an use a GAN. Note that the
    model, i.e. the architecture of the network is not handled by this class.
    """
    def default_params(self):


        # Global parameters
        # -----------------
        d_param = super().default_params()

         # Optimization parameters
        # -----------------------
        d_opt = dict()
        d_opt['optimizer'] = "rmsprop"
        d_opt['learning_rate'] = 3e-5
        d_opt['kwargs'] = dict()
        d_param['optimization'] = dict()
        d_param['optimization']['discriminator'] = deepcopy(d_opt)
        d_param['optimization']['generator'] = deepcopy(d_opt)
        d_param['optimization']['encoder'] = deepcopy(d_opt)
        d_param['optimization']['n_critic'] = 5
        d_param['optimization']['epoch'] = 10
        d_param['optimization']['batch_size'] = 8
        d_param['Nstats'] = None

        return d_param


    def __init__(self,  model, params=None, name=None):
        """Build the GAN network.

        Input arguments
        ---------------
        * params : structure of parameters
        * model  : model class for the architecture of the network

        Please refer to the module `model` for details about
        the requirements of the class model.
        """
        if params is None:
            params = {}
        super().__init__(model=model, params=params, name=name)
  

    def _add_optimizer(self):

        if self._net.has_encoder:
            varsuffixes = ['discriminator', 'generator', 'encoder']
        else:
            varsuffixes = ['discriminator', 'generator']

        losses = self._net.loss
        t_vars = tf.trainable_variables()
        self._optimize = []

        # global_step = tf.Variable(0, name="global_step", trainable=False)
        print('\nBuild the optimizers: ')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            for index, varsuffix in enumerate(varsuffixes):
                print(' * {} '.format(varsuffix))
                s_vars = [var for var in t_vars if varsuffix in var.name]
                params = self.params['optimization'][varsuffix]
                print(yaml.dump(params))
                optimizer = self.build_optmizer(params)
                grads_and_vars = optimizer.compute_gradients(losses[index], var_list=s_vars)
                apply_opt = optimizer.apply_gradients(grads_and_vars)
                self._optimize.append(tf.group(apply_opt, *self.net.constraints))

                # Summaries
                grad_norms = [tf.nn.l2_loss(g[0])*2 for g in grads_and_vars]
                grad_norm = [tf.reduce_sum(grads) for grads in grad_norms]
                final_grad = tf.sqrt(tf.reduce_sum(grad_norm))
                tf.summary.scalar(varsuffix+"/Gradient_Norm", final_grad, collections=["train"])
                #if self.params['optimization'][varsuffix]['optimizer'] == 'adam':
                #    beta1_power, beta2_power = optimizer._get_beta_accumulators()
                #    learning_rate = self.params['optimization'][varsuffix]['learning_rate']
                #    optim_learning_rate = learning_rate*(tf.sqrt(1 - beta2_power) /(1 - beta1_power))
                #    tf.summary.scalar(varsuffix+'/ADAM_learning_rate', optim_learning_rate, collections=["train"])

    @staticmethod
    def build_optmizer(params):

        learning_rate = params['learning_rate']
        optimizer = params['optimizer']
        kwargs = params['kwargs']

        if optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                **kwargs)
        elif optimizer == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate=learning_rate, **kwargs)
        elif optimizer == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate, **kwargs)
        else:
            raise Exception(" [!] Choose optimizer between [adam,rmsprop,sgd]")

        return optimizer

    def train(self, dataset, **kwargs):
        if self.params['Nstats']:
            assert(dataset.N>=self.params['Nstats'])
            self.summary_dataset = itertools.cycle(dataset.iter(self.params['Nstats']))
            self.net.preprocess_summaries(dataset.get_samples(self.params['Nstats']), rerun=False)
        super().train(dataset, **kwargs)

    def _run_optimization(self, feed_dict, idx):
        # Update discriminator
        for _ in range(self.params['optimization']['n_critic']):
            _, loss_d = self._sess.run([self._optimize[0], self.net.loss[0]], feed_dict=feed_dict)
            # Small hack: redraw a new latent variable
            feed_dict[self.net.z] = self.net.sample_latent(self.params['optimization']['batch_size'])
        # Update Encoder
        if self.net.has_encoder:
            _, loss_e = self._sess.run(
                self._optimize[2], self.net.loss[2],
                feed_dict=feed_dict)
        # Update generator
        curr_loss = self._sess.run([self._optimize[1], *self.net.loss], feed_dict=feed_dict)[1:]
        if idx == 0:
            self._epoch_loss_disc = 0
            self._epoch_loss_gen = 0
        self._epoch_loss_disc += curr_loss[0]
        self._epoch_loss_gen += curr_loss[1]
        return curr_loss

    def _train_log(self, feed_dict):
        super()._train_log(feed_dict)
        X_real, X_fake = self._sess.run([self.net.X_real, self.net.X_fake], feed_dict=feed_dict)
        if self.params['Nstats']:
            sum_batch = next(self.summary_dataset)
            sum_feed_dict = self._get_dict(**self._net.batch2dict(sum_batch))
            X_fake = self._generate_sample_safe(is_feed_dict=True, feed_dict=sum_feed_dict)
            X_real = self._sess.run(self.net.X_real, feed_dict=sum_feed_dict)
            feed_dict = self.net.compute_summaries(X_real, X_fake, feed_dict)
        else:
            feed_dict = self.net.compute_summaries(X_real, X_fake, feed_dict)

        summary = self._sess.run(self.net.summary, feed_dict=feed_dict)
        self._summary_writer.add_summary(summary, self._counter)

    def _print_log(self, idx, curr_loss):
        current_time = time.time()
        batch_size = self.params['optimization']['batch_size']
        print(" * Epoch: [{:2d}] [{:4d}/{:4d}] "
              "Counter:{:2d}\t"
              "({:4.1f} min\t"
              "{:4.3f} examples/sec\t"
              "{:4.2f} sec/batch)\n"
              "   Disc batch loss:{:.8f}\t"
              "Disc epoch loss:{:.8f}\n"
              "   Gen batch loss:{:.8f}\t"
              "Gen epoch loss:{:.8f}".format(
                  self._epoch, 
                  idx+1, 
                  self._n_batch,
                  self._counter,
                  (current_time - self._time['start_time']) / 60,
                  self._params['print_every'] * batch_size / (current_time - self._time['prev_iter_time']),
                  (current_time - self._time['prev_iter_time']) / self._params['print_every'],
                  curr_loss[0],
                  self._epoch_loss_disc/(idx+1),
                  curr_loss[1],
                  self._epoch_loss_gen/(idx+1)))
        self._time['prev_iter_time'] = current_time

    def generate(self, N=None, sess=None, checkpoint=None, **kwargs):
        """Generate new samples.

        The user can chose between different options depending on the model.

        **kwargs contains all possible optional arguments defined in the model.

        Arguments
        ---------
        * N : number of sample (Default None)
        * sess : tensorflow Session (Default None)
        * checkpoint : number of the checkpoint (Default None)
        * kwargs : keywords arguments that are defined in the model
        """
        if N is None and len(kwargs)==0:
            raise ValueError("Please Specify N or variable for the models")
        dict_latent = dict()
        if not('z' in kwargs.keys()):
            print('Sampling z')
            if N is None:
                N = len(kwargs[list(kwargs.keys())[0]])
            dict_latent['z'] =self.net.sample_latent(N)
        if sess is not None:
            self._sess = sess
            # print("Not loading a checkpoint")

        else:
            self._sess = tf.Session()
            res = self.load(checkpoint=checkpoint)
            if res:
                print("Checkpoint succesfully loaded!")
        samples = self._generate_sample_safe(**dict_latent, **kwargs)
        if sess is None:
            self._sess.close()
        return samples

    # TODO: move this to the nnsystem class
    def _special_vstack(self, gi):
        if type(gi[0]) is np.ndarray:
            return np.vstack(gi)
        else:
            s = []
            for j in range(len(gi[0])):
                s.append(np.vstack([el[j] for el in gi]))
            return tuple(s)

    # TODO: move this to the nnsystem class
    def _generate_sample_safe(self, is_feed_dict=False, **kwargs):
        gen_images = []
        if is_feed_dict:
            feed_dict = kwargs['feed_dict']
            N = len(feed_dict[list(feed_dict.keys())[0]])
        else:
            N = len(kwargs[list(kwargs.keys())[0]])
        sind = 0
        bs = self.params['optimization']['batch_size']
        if N > bs:
            nb = (N - 1) // bs
            for i in range(nb):
                if is_feed_dict:
                    feed_dict = self._slice_feed_dict(index=slice(sind, sind + bs), **kwargs)
                else:
                    feed_dict = self._get_dict(index=slice(sind, sind + bs), **kwargs)
                gi = self._sess.run(self._net.outputs, feed_dict=feed_dict)
                gen_images.append(gi)
                sind = sind + bs
        if is_feed_dict:
            feed_dict = self._slice_feed_dict(index=slice(sind, N), **kwargs)
        else:
            feed_dict = self._get_dict(index=slice(sind, N), **kwargs)
        gi = self._sess.run(self._net.outputs, feed_dict=feed_dict)
        gen_images.append(gi)
        return self._special_vstack(gen_images)


class DualGANsystem(GANsystem):

    def _classical_gan_loss_d(self, real, fake, epsilon=0):
        return -tf.reduce_mean(tf.log(epsilon + tf.nn.sigmoid(real)) + tf.log(epsilon +(1-tf.nn.sigmoid(fake))))
        # return -tf.reduce_mean(tf.log(tf.nn.sigmoid(real)) - fake + tf.log(tf.nn.sigmoid(fake)))

    def _classical_gan_loss_g(self, fake, epsilon=1e-5):
        return tf.reduce_mean(tf.log(epsilon + (1-tf.nn.sigmoid(fake))))
        # return tf.reduce_mean(- fake + tf.log(tf.nn.sigmoid(fake)))


    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.plc_float = tf.placeholder(tf.float32)
        self.plc_float_r = tf.placeholder(tf.float32)
        self.disc_loss_calc2 = tf.reduce_mean(self.plc_float_r - self.plc_float)

        with tf.variable_scope('worst_calc', reuse=tf.AUTO_REUSE):
            new_opt = tf.train.RMSPropOptimizer(learning_rate=3e-5)
            self.df = self.net.discriminator(self._net.X_fake, reuse=tf.AUTO_REUSE, scope="TMPdisc")
            self.dr = self.net.discriminator(self._net.X_real, reuse=tf.AUTO_REUSE, scope="TMPdisc")
            # disc_loss_worst = -tf.reduce_mean(self.dr - self.df)
            disc_loss_worst = self._classical_gan_loss_d(self.dr, self.df)
            t_vars = tf.global_variables()
            d_vars_worst = [var for var in t_vars if 'TMPdisc' in var.name]
            self.find_worst_d = new_opt.minimize(disc_loss_worst, var_list=d_vars_worst)


        with tf.variable_scope('worst_calc_gen', reuse=tf.AUTO_REUSE):
            new_opt_gen = tf.train.RMSPropOptimizer(learning_rate=3e-5)
            x_w = self.net.generator(self._net.z, reuse=False, scope="TMPgen")
        self.df_w = self.net.discriminator(x_w, reuse=True, scope="discriminator")
        self.dr_w = self.net.discriminator(self._net.X_real, reuse=True, scope="discriminator")

        with tf.variable_scope('worst_calc_gen', reuse=tf.AUTO_REUSE):
            # gen_loss_worst = tf.reduce_mean(self.dr_w - self.df_w)
            # gen_loss_worst = self._classical_gan_loss_g(self.df_w)
            gen_loss_worst = - self._classical_gan_loss_d(self.dr_w, self.df_w)
            
            t_vars = tf.global_variables()
            g_vars_worst = [var for var in t_vars if 'TMPgen' in var.name]
            self.find_worst_g = new_opt_gen.minimize(gen_loss_worst, var_list=g_vars_worst)

        t_vars = tf.global_variables()
        d_init = [var for var in t_vars if 'worst_calc' in var.name]
        self.init_new_vars_op = tf.initialize_variables(d_init)

        curr_to_tmp = []
        t_vars = tf.global_variables()
        d_vars_tmp = [var for var in t_vars if 'TMPdisc' in var.name and 'RMSProp' not in var.name]
        d_vars_0 = [var for var in t_vars if 'discriminator/' in var.name and 'RMSProp' not in var.name]
        g_vars_tmp = [var for var in t_vars if 'TMPgen' in var.name and 'RMSProp' not in var.name]
        g_vars_0 = [var for var in t_vars if 'generator/' in var.name and 'RMSProp' not in var.name]
        for j in range(0, len(d_vars_tmp)):
            print (d_vars_tmp[j])
            curr_to_tmp.append(d_vars_tmp[j].assign(d_vars_0[j]))
        for j in range(0, len(g_vars_tmp)):
            curr_to_tmp.append(g_vars_tmp[j].assign(g_vars_0[j]))

        self.current_to_tmp = tf.group(*curr_to_tmp)

        # This is not clean. It should probably done at the model level.    
        self.dualitygap_score_pl = tf.placeholder(tf.float32, name='duality/gap')
        tf.summary.scalar('duality/gap', self.dualitygap_score_pl, collections=['train'])

        self.worst_minmax_pl = tf.placeholder(tf.float32, name='duality/minmax')
        tf.summary.scalar('duality/minmax', self.worst_minmax_pl, collections=['train'])

        self.worst_maxmin_pl = tf.placeholder(tf.float32, name='duality/maxmin')
        tf.summary.scalar('duality/maxmin', self.worst_maxmin_pl, collections=['train'])
        self._summaries = tf.summary.merge(tf.get_collection("train"))

    def calculate_metrics(self):

        if self._sess is None:
            self._sess = tf.Session()
            sess_to_be_closed = True
        else:
            sess_to_be_closed = False

        # First randomly initialize the new variables for the optimization of the new D_tmp/G_tmp
        self._sess.run(self.init_new_vars_op)
        # Assign the weights to the new D_tmp/G_tmp to be the those of the current D/G
        self._sess.run(self.current_to_tmp)

        # for fixed G, find the worst D_tmp
        # This need to be replaced with an infinite iterator define in the train function.
        for j in range(0, 500):
            batch_curr = next(self.paulina_dataset_iter)
            feed_dict = self._get_dict(**self._net.batch2dict(batch_curr))
            self._sess.run(self.find_worst_d, feed_dict=feed_dict)
        # calculate the worst minmax
        feed_dict = self._get_dict(**self._net.batch2dict(batch_curr))
        df_final = self._sess.run(self.df, feed_dict=feed_dict) # here you need to feed z
        dr_final = self._sess.run(self.dr, feed_dict=feed_dict)
        worst_minmax = self._sess.run(self.disc_loss_calc2, feed_dict={self.plc_float: df_final, self.plc_float_r: dr_final})

        for j in range(0, 500):
            batch_curr = next(self.paulina_dataset_iter)
            feed_dict = self._get_dict(**self._net.batch2dict(batch_curr))
            self._sess.run(self.find_worst_g, feed_dict=feed_dict)
        # calculate the worst maxmin
        feed_dict = self._get_dict(**self._net.batch2dict(batch_curr))
        df_final = self._sess.run(self.df_w, feed_dict=feed_dict)
        dr_final = self._sess.run(self.dr_w, feed_dict=feed_dict)
        worst_maxmin = self._sess.run(self.disc_loss_calc2, feed_dict={self.plc_float: df_final, self.plc_float_r: dr_final})

        # report the metrics
        dualitygap_score = worst_minmax - worst_maxmin
        print ('The duality gap score is: ')
        print ('{0:.16f}'.format(dualitygap_score))
        print ('The minmax is: ')
        print ('{0:.16f}'.format(worst_minmax))

        if sess_to_be_closed:
            self._sess.close()
            self._sess = None
        return (dualitygap_score, worst_minmax, worst_maxmin)

    def train(self, dataset, **kwargs):
        batch_size = self.params['optimization']['batch_size']

        self.paulina_dataset_iter = itertools.cycle(dataset.iter(batch_size))
        super().train(dataset, **kwargs)

    def _train_log(self, feed_dict):
        (dualitygap_score, worst_minmax, worst_maxmin) = self.calculate_metrics()
        feed_dict[self.dualitygap_score_pl] = dualitygap_score
        feed_dict[self.worst_minmax_pl] = worst_minmax
        feed_dict[self.worst_maxmin_pl] = worst_maxmin
        super()._train_log(feed_dict)

class UpscaleGANsystem(GANsystem):

    def upscale_image(self, N=None, small=None, resolution=None, checkpoint=None, sess=None):
        """Upscale image using the lappachsimple model, or upscale_WGAN_pixel_CNN model.

        For model upscale_WGAN_pixel_CNN, pass num_samples to generate and resolution of the final bigger histogram.
        for model lappachsimple         , pass small.

        3D only works for upscale_WGAN_pixel_CNN.

        This function can be accelerated if the model is created only once.
        """
        # Number of sample to produce
        if small is not None:
            N = small.shape[0]
        if N is None:
            raise ValueError('Please specify small or N.')
        if self.net.data_size==1:
            raise NotImplementedError()
        # Output dimension of the generator
        soutx, souty = self.params['net']['shape'][:2]
        if self.net.data_size==3:
            soutz = self.params['net']['shape'][2]

        if small is not None:
            # Dimension of the low res image
            lx, ly = small.shape[1:3]
            if self.net.data_size==3:
                lz = small.shape[3]

            # Input dimension of the generator
            sinx = soutx // self.params['net']['upscaling']
            siny = souty // self.params['net']['upscaling']
            if self.net.data_size==3:
                sinz = soutz // self.params['net']['upscaling']

            # Number of part to be generated
            nx = lx // sinx
            ny = ly // siny
            if self.net.data_size==3:
                nz = lz // sinz

        else:
            sinx = siny = sinz = None
            if resolution is None:
                raise ValueError("Both small and resolution cannot be None")
            else:
                nx = resolution // soutx
                ny = resolution // souty
                if self.net.data_size==3:
                    nz = resolution // soutz

        # If no session passed, create a new one and load a checkpoint.
        if sess is None:
            new_sess = tf.Session()
            res = self.load(sess=new_sess, checkpoint=checkpoint)
            if res:
                print('Checkpoint successfully loaded!')
        else:
            new_sess = sess

        if self.net.data_size==3:
            output_image = self.generate_3d_output(new_sess, N, nx, ny, nz, soutx, souty, soutz, small, sinx, siny, sinz)
        else:
            output_image = self.generate_2d_output(new_sess, N, nx, ny, soutx, souty, small, sinx, siny)

        # If a new session was created, close it. 
        if sess is None:
            new_sess.close()

        return np.squeeze(output_image)
    def generate_3d_output(self, sess, N, nx, ny, nz, soutx, souty, soutz, small,
                           sinx, siny, sinz):
        # this function does only support 1 channel image
        output_image = np.zeros(
            shape=[N, soutz * nz, souty * ny, soutx * nx, 1], dtype=np.float32)
        output_image[:] = np.nan

        print('Total number of patches = {}*{}*{} = {}'.format(
            nx, ny, nz, nx * ny * nz))

        for k in range(nz):  # height
            for j in range(ny):  # rows
                for i in range(nx):  # columns

                    # 1) Generate the border
                    border = np.zeros([N, soutz, souty, soutx, 7])

                    if j:  # one row above, same height
                        border[:, :, :, :, 0:1] = output_image[:,
                            k * soutz:(k + 1) * soutz,
                            (j - 1) * souty:j * souty,
                            i * soutx:(i + 1) * soutx, :]
                    if i:  # one column left, same height
                        border[:, :, :, :, 1:2] = output_image[:,
                            k * soutz:(k + 1) * soutz,
                            j * souty:(j + 1) * souty,
                            (i - 1) *soutx:i * soutx, :]
                    if i and j:  # one row above, one column left, same height
                        border[:, :, :, :, 2:3] = output_image[:,
                            k * soutz:(k + 1) * soutz,
                            (j - 1) * souty:j * souty,
                            (i - 1) * soutx:i * soutx, :]
                    if k:  # same row, same column, one height above
                        border[:, :, :, :, 3:4] = output_image[:,
                            (k - 1) * soutz:k * soutz,
                            j * souty:(j + 1) * souty,
                            i * soutx:(i + 1) * soutx, :]
                    if k and j:  # one row above, same column, one height above
                        border[:, :, :, :, 4:5] = output_image[:,
                            (k - 1) * soutz:k * soutz,
                            (j - 1) * souty:j *souty,
                            i * soutx:(i + 1) * soutx, :]
                    if k and i:  # same row, one column left, one height above
                        border[:, :, :, :, 5:6] = output_image[:,
                            (k - 1) * soutz:k * soutz,
                            j * souty:(j + 1) * souty,
                            (i - 1) * soutx:i * soutx, :]
                    if k and i and j:  # one row above, one column left, one height above
                        border[:, :, :, :, 6:7] = output_image[:,
                            (k - 1) * soutz:k * soutz,
                            (j - 1) * souty:j * souty,
                            (i - 1) * soutx:i * soutx, :]

                    # 2) Prepare low resolution
                    if small is not None:
                        downsampled = small[:, 
                                            k * sinz:(k + 1) * sinz,
                                            j * siny:(j + 1) * siny,
                                            i * sinx:(i + 1) * sinx, :]
                    else:
                        downsampled = None

                    # 3) Generate the image
                    print('Current patch: column={}, row={}, height={}'.format(
                        i + 1, j + 1, k + 1))
                    if downsampled is not None:
                        gen_sample = self.generate(N=N, borders=border, X_down=downsampled, sess=sess)
                    else:
                        gen_sample = self.generate(N=N, borders=border, sess=sess)
                    output_image[:,
                        k * soutz:(k + 1) * soutz,
                        j * souty:(j + 1) *souty,
                        i * soutx:(i + 1) * soutx, :] = gen_sample

        return output_image


    def generate_2d_output(self, sess, N, nx, ny, soutx, souty, small, sinx,
                           siny):
#         nc = self.net.params['shape'][-1]//4 # number of channel for the image
        nc = 1
        output_image = np.zeros(
            shape=[N, soutx * nx, souty * ny, nc], dtype=np.float32)
        output_image[:] = np.nan
        
        for j in range(ny):
            for i in range(nx):
                # 1) Generate the border
                border = np.zeros([N, soutx, souty, 3*nc])
                if i:
                    border[:, :, :, :nc] = output_image[:, (i - 1) * soutx:i * soutx, j * souty:(j + 1) * souty, :]
                if j:
                    border[:, :, :, nc:2*nc] = output_image[:, i * soutx:(i + 1) * soutx, (j - 1) * souty:j * souty, :]
                if i and j:
                    border[:, :, :, 2*nc:3*nc] = output_image[:, (i - 1) * soutx:i * soutx, (j - 1) * souty:j * souty, :]

                if small is not None:
                    # 2) Prepare low resolution
                    print(small.shape)
                    downsampled = small[:N][:, i * sinx:(i + 1) * sinx, j * siny:(j + 1) * siny]
                else:
                    downsampled = None

                # 3) Generate the image
                print('Current patch: column={}, row={}'.format(j + 1, i + 1))
                if  downsampled is not None:
                    gen_sample = self.generate(N=N, borders=border, X_down=downsampled, sess=sess)
                else:
                    gen_sample = self.generate(N=N, borders=border, sess=sess)

                output_image[:, i * soutx:(i + 1) * soutx, j * souty:(j + 1) *
                             souty, :] = gen_sample

        return output_image


