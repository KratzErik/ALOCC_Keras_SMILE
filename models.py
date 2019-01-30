# In this file, the autoencoder and discriminator models are constructed and trained.
# Architectures can either be coded explicitly here, by editing the functions:
#       build_generator() for the autoencoder
#       build_discriminator() for the discriminator
# or more easily defined using the architecture options in ./configuration.py

# The optimization (loss) objectives are setup in build_model() and the models are trained
# in function train(). The train() function calls several utility functions in this file,
# e.g. for saving and loading checkpoints.

# When the file is run as a script, the code at the bottom of the file is executed. This 
# will create a configuration object from the specified command line arguments dataset and 
# exp_name, and do the following:

# 1. Create an ALOCC model
# 2. Check if there exists a model checkpoint matching the specified number of epochs
# 3. If more epochs are needed (includes the case were no checkpoint exists), train the
#    model until the specified number of epochs is completed.
    
#    The model will be checkpointed at intervals specified in ./configuration.py, and at 
#    the final epoch.


from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import Concatenate
from keras.layers import MaxPool2D as MaxPool
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from keras import losses
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
import keras.backend as K
import scipy
import logging
import matplotlib.pyplot as plt
import os
from ae_architecture import AE_Architecture
from d_architecture import D_Architecture
from configuration import Configuration
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import datetime
from utils import *
from kh_tools import *

from shutil import copyfile
import argparse

class ALOCC_Model():
    def __init__(self,
               input_height=28,input_width=28, output_height=28, output_width=28,
               attention_label=1, is_training=True,
               z_dim=100, gf_dim=64, df_dim=64, c_dim=3,
               dataset_name=None, dataset_address=None, input_fname_pattern=None,
               log_dir='./log', r_alpha = 0.2,
               kb_work_on_patch=True, nd_patch_size=(10, 10), n_stride=1,
               n_fetch_data=10, outlier_dir = '', experiment_name = '', cfg=None):
        """
        This is the main class of our Adversarially Learned One-Class Classifier for Novelty Detection.
        :param sess: TensorFlow session.
        :param input_height: The height of image to use.
        :param input_width: The width of image to use.
        :param output_height: The height of the output images to produce.
        :param output_width: The width of the output images to produce.
        :param attention_label: Conditioned label that growth attention of training label [1]
        :param is_training: True if in training mode.
        :param z_dim:  (optional) Dimension of dim for Z, the output of encoder. [100]
        :param gf_dim: (optional) Dimension of gen filters in first conv layer, i.e. g_decoder_h0. [16] 
        :param df_dim: (optional) Dimension of discrim filters in first conv layer, i.e. d_h0_conv. [16] 
        :param c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        :param dataset_name: 'UCSD', 'mnist' or custom defined name.
        :param dataset_address: path to dataset folder. e.g. './dataset/mnist'.
        :param input_fname_pattern: Glob pattern of filename of input images e.g. '*'.
        :param log_dir: log directory for checkpoints, training diagnostics and test results
        :param r_alpha: Refinement parameter, trade-off hyperparameter for the G network loss to reconstruct input images. [0.2]
        :param kb_work_on_patch: Boolean value for working on PatchBased System or not, only applies to UCSD dataset [True]
        :param nd_patch_size:  Input patch size, only applies to UCSD dataset.
        :param n_stride: PatchBased data preprocessing stride, only applies to UCSD dataset.
        :param n_fetch_data: Fetch size of Data, only applies to UCSD dataset. 
        """

        if cfg is None:
            print("ERROR: No configuration for ALOCC model.")
        else:
            self.cfg = cfg
        self.b_work_on_patch = kb_work_on_patch

        # Create different log dirs
        self.log_dir = self.cfg.log_dir
        self.train_dir = self.log_dir + 'train/'
        self.checkpoint_dir = self.log_dir + 'models/'
        self.test_dir = self.log_dir + 'test/'

        self.is_training = is_training
        self.r_alpha = r_alpha

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.dataset_name = dataset_name
        self.dataset_address= dataset_address
        self.input_fname_pattern = input_fname_pattern

        self.outlier_dir = outlier_dir

        self.attention_label = attention_label

        self.experiment_name = experiment_name
        self.start_epoch = 0

        if self.cfg.hardcoded_architecture == 'ALOCC_mnist':
            print("Using original ALOCC architectures")
            self.ae_architecture = None
            self.d_architecture = None
        else:
            self.ae_architecture = AE_Architecture(cfg = self.cfg)
            self.d_architecture = D_Architecture(cfg = self.cfg)
        if self.is_training:
          logging.basicConfig(filename='ALOCC_loss.log', level=logging.INFO)

        if self.dataset_name == 'mnist':
          (X_train, y_train), (_, _) = mnist.load_data()
          print("Loaded mnist data")
          # Make the data range between 0~1.
          X_train = X_train / 255
          specific_idx = np.where(y_train == self.attention_label)[0]
          inlier_data = X_train[specific_idx].reshape(-1, 28, 28, 1)
          if self.is_training:
              self.data = inlier_data
          else: # load test data
               X_test_in = inlier_data[np.random.choice(len(inlier_data), self.cfg.n_test_in, replace=False)]
               n_test_out = self.cfg.n_test - self.cfg.n_test_in
               outlier_idx = np.where(y_train != self.attention_label)[0]
               outlier_data = X_train[outlier_idx].reshape(-1, 28, 28, 1)
               X_test_out = outlier_data[np.random.choice(len(outlier_data), n_test_out, replace=False)]
               self.data = np.concatenate([X_test_in, X_test_out])
               y_test_in  = np.ones((len(X_test_in),),dtype=np.int32)
               y_test_out = np.zeros((len(X_test_out),),dtype=np.int32)
               self.test_labels = np.concatenate([y_test_in, y_test_out])

          self.c_dim = 1

        elif self.dataset_name in ('dreyeve','prosivic'):
            self.c_dim = 3
            if self.is_training:
                X_train = np.array([img_to_array(load_img(self.cfg.train_folder + filename)) for filename in os.listdir(self.cfg.train_folder)][:self.cfg.n_train])
                self.data = X_train / 255.0
            else: #load test data     
                n_test_out = self.cfg.n_test - self.cfg.n_test_in
                X_test_in = np.array([img_to_array(load_img(self.cfg.test_in_folder + filename)) for filename in os.listdir(self.cfg.test_in_folder)][:self.cfg.n_test_in])
                X_test_out = np.array([img_to_array(load_img(self.outlier_dir + filename)) for filename in os.listdir(self.outlier_dir)][:n_test_out])
                y_test_in  = np.ones((len(X_test_in),),dtype=np.int32)
                y_test_out = np.zeros((len(X_test_out),),dtype=np.int32)
                self.data = np.concatenate([X_test_in, X_test_out]) / 255.0
                self.test_labels = np.concatenate([y_test_in, y_test_out])
        else:
          assert('Error in loading dataset')

        self.grayscale = (self.c_dim == 1)
        self.build_model()

        # Print dataset size
        if self.is_training:
            print("Training set size: ", len(self.data))
        else:
            print("Test set:\n\tInliers: %d\n\tOutliers: %d"%(len(X_test_in), len(X_test_out)))

    def build_generator(self, input_shape):
        """Build the generator/R network.
        
        Arguments:
            input_shape {list} -- Generator input shape.
        
        Returns:
            [Tensor] -- Output tensor of the generator/R network.
        """

        if self.ae_architecture is None:
            image = Input(shape=input_shape, name='z')
            # Encoder.
            x = Conv2D(filters=self.df_dim * 2, kernel_size = 5, strides=2, padding='same', name='g_encoder_h0_conv')(image)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=self.cfg.lrelu_alpha)(x)
            x = Conv2D(filters=self.df_dim * 4, kernel_size = 5, strides=2, padding='same', name='g_encoder_h1_conv')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=self.cfg.lrelu_alpha)(x)
            x = Conv2D(filters=self.df_dim * 8, kernel_size = 5, strides=2, padding='same', name='g_encoder_h2_conv')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=self.cfg.lrelu_alpha)(x)

            # Decoder.
            x = Conv2DTranspose(self.gf_dim*2, kernel_size = 5, strides=2, activation='relu', padding='same', output_padding=0, name='g_decoder_h0')(x)
            x = BatchNormalization()(x)
            x = Conv2DTranspose(self.gf_dim*1, kernel_size = 5, strides=2, activation='relu', padding='same', output_padding=1, name='g_decoder_h1')(x)
            x = BatchNormalization()(x)
            x = Conv2DTranspose(self.c_dim,    kernel_size = 5, strides=2, activation='tanh', padding='same', output_padding=1, name='g_decoder_h2')(x)

            return Model(image, x, name='R')
        
        else: # architecture built from ae_architecture object
            # This code block reads autoencoder architecture settings from ./configuration object cfg, 
            # and adds layers in loops
            image = Input(shape=input_shape, name='z')
            x = image
            # Encoder
            for m in range(self.ae_architecture.n_conv_modules): # Loop over conv modules
                k_size = self.ae_architecture.filter_size[m]
                stride = self.ae_architecture.stride[m]
                channels = self.ae_architecture.channels[m]
                for l in range(self.ae_architecture.n_conv_layers_per_module[m]):
                    if l == self.ae_architecture.n_conv_layers_per_module[m] - 1: # last layer in module
                        stride = self.ae_architecture.dim_red_stride[m] # sets stride to 2 in last layer if max_pool = False
                    x = Conv2D(filters=channels, kernel_size = k_size, strides=stride, padding='same', name='g_encoder_h%d_%d_conv'%(m,l))(x)
                    if self.ae_architecture.use_batch_norm:
                        x = BatchNormalization()(x)
                    if self.ae_architecture.use_dropout:
                        x = Dropout(self.ae_architecture.dropout_rate)(x)
                    x = LeakyReLU(alpha=self.cfg.lrelu_alpha)(x)
                if self.ae_architecture.max_pool:
                    x = MaxPool(self.ae_architecture.pool_size[m])(x)

            if self.ae_architecture.n_dense_layers > 0:
                # Compute current image dimensions
                height_scale_factor = np.prod(self.ae_architecture.dim_red_stride)
                height_before_dense = input_shape[0]//height_scale_factor
                width_scale_factor = np.prod(self.ae_architecture.dim_red_stride)
                width_before_dense = input_shape[1]//width_scale_factor
                channels_before_dense = self.ae_architecture.channels[-1]
                shape_before_dense = [height_before_dense,width_before_dense,channels_before_dense]

                x = Flatten()(x)
                for d in range(self.ae_architecture.n_dense_layers):
                    x = Dense(self.ae_architecture.n_dense_units[d], name='g_encoder_h%d_lin'%d)(x)
                    if self.ae_architecture.use_batch_norm:
                        x = BatchNormalization()(x)
                    if self.ae_architecture.use_dropout:
                        x = Dropout(self.ae_architecture.dropout_rate)(x)
                    x = LeakyReLU(alpha=self.cfg.lrelu_alpha)(x)

            # Decoder
            if self.ae_architecture.n_dense_layers > 0:
                n_dense_units_flip = np.flip(self.ae_architecture.n_dense_units[:-1])
                n_dense_units_flip = np.append(n_dense_units_flip,np.prod(shape_before_dense,dtype=int))

                for d in range(self.ae_architecture.n_dense_layers):
                    x = Dense(int(n_dense_units_flip[d]), name='g_decoder_h%d_lin'%d)(x)
                    if self.ae_architecture.use_batch_norm:
                        x = BatchNormalization()(x)
                    if self.ae_architecture.use_dropout:
                        x = Dropout(self.ae_architecture.dropout_rate)(x)
                    x = LeakyReLU(alpha=self.cfg.lrelu_alpha)(x)

                x = Reshape(shape_before_dense)(x)

            n_conv_layers_per_module_flip = np.flip(self.ae_architecture.n_conv_layers_per_module)
            channels_flip = np.flip(self.ae_architecture.channels[:-1])
            channels_flip = np.append(channels_flip, self.c_dim)
            filter_size_flip = np.flip(self.ae_architecture.filter_size)
            stride_flip = np.flip(self.ae_architecture.stride)

            for m in range(self.ae_architecture.n_conv_modules): # Loop over conv modules
                channels = channels_flip[m]
                if self.ae_architecture.max_pool:
                    x = UpSampling2D((self.ae_architecture.pool_size,self.ae_architecture.pool_size))(x)
                for l in range(n_conv_layers_per_module_flip[m]):
                    is_output_layer = (m == self.ae_architecture.n_conv_modules - 1 and l == self.ae_architecture.n_conv_layers_per_module[m]-1)
                    stride = int(stride_flip[m])
                    k_size = int(filter_size_flip[m])
                    if l == 0: # last layer in module
                        stride = self.ae_architecture.dim_red_stride[m] # sets stride to 2 in last layer if max_pool = False
                    if self.ae_architecture.max_pool:
                        x = Conv2DTranspose(filters=channels, kernel_size = k_size, strides=stride, padding='same',  name='g_decoder_h%d_%d_conv'%(m,l))(x)
                    else:
                        outpad = int((k_size-stride)%2)
                        x = Conv2DTranspose(filters=channels, kernel_size = k_size, strides=stride, padding='same', name='g_decoder_h%d_%d_conv'%(m,l))(x)
                        if self.ae_architecture.use_batch_norm and (self.ae_architecture.output_batch_norm or not is_output_layer):
                            x = BatchNormalization()(x)
                    if self.ae_architecture.use_dropout:
                        x = Dropout(self.ae_architecture.dropout_rate)(x)
                    if is_output_layer:
                        x = Activation('sigmoid')(x)
                    else:
                        x = LeakyReLU(alpha=self.cfg.lrelu_alpha)(x)
            
            return Model(image, x, name='R')
                
    
    def build_discriminator(self, input_shape):
        """Build the discriminator/D network
        
        Arguments:
            input_shape {list} -- Input tensor shape of the discriminator network, either the real unmodified image
                or the generated image by generator/R network.
        
        Returns:
            [Tensor] -- Network output tensors.
        """
        if self.d_architecture is None:
            image = Input(shape=input_shape, name='d_input')
            x = Conv2D(filters=self.df_dim, kernel_size = 5, strides=2, padding='same', name='d_h0_conv')(image)
            x = LeakyReLU(alpha=self.cfg.lrelu_alpha)(x)

            x = Conv2D(filters=self.df_dim*2, kernel_size = 5, strides=2, padding='same', name='d_h1_conv')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=self.cfg.lrelu_alpha)(x)

            x = Conv2D(filters=self.df_dim*4, kernel_size = 5, strides=2, padding='same', name='d_h2_conv')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=self.cfg.lrelu_alpha)(x)

            x = Conv2D(filters=self.df_dim*8, kernel_size = 5, strides=2, padding='same', name='d_h3_conv')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=self.cfg.lrelu_alpha)(x)

            x = Flatten()(x)
            x = Dense(1, activation='sigmoid', name='d_h3_lin')(x)

            return Model(image, x, name='D')
        
        else: # architecture built from d_architecture object
            # This code block reads discriminator architecture settings from ./configuration object cfg, 
            # and adds layers in loops
            image = Input(shape=input_shape, name='z')
            x = image
            
            # Encoder
            for m in range(self.d_architecture.n_conv_modules): # Loop over conv modules
                k_size = self.d_architecture.filter_size[m]
                stride = self.d_architecture.stride[m]
                channels = self.d_architecture.channels[m]
                for l in range(self.d_architecture.n_conv_layers_per_module[m]):
                    if l == self.d_architecture.n_conv_layers_per_module[m] - 1: # last layer in module
                        stride = self.d_architecture.dim_red_stride[m] # sets stride to 2 in last layer if max_pool = False
                    x = Conv2D(filters=channels, kernel_size = k_size, strides=stride, padding='same', name='d_h%d_%d_conv'%(m,l))(x)
                    if self.d_architecture.use_batch_norm:
                        x = BatchNormalization()(x)
                    if self.d_architecture.use_dropout:
                        x = Dropout(self.d_architecture.dropout_rate)(x)
                    x = LeakyReLU(alpha=self.cfg.lrelu_alpha)(x)
                if self.d_architecture.max_pool:
                    x = MaxPool(self.d_architecture.pool_size[m])(x)

            # Compute current image dimensions
            height_scale_factor = np.prod(self.d_architecture.dim_red_stride)
            height_before_dense = input_shape[0]//height_scale_factor
            width_scale_factor = np.prod(self.d_architecture.dim_red_stride)
            width_before_dense = input_shape[1]//width_scale_factor
            channels_before_dense = self.d_architecture.channels[-1]
            shape_before_dense = [height_before_dense,width_before_dense,channels_before_dense]

            x = Flatten()(x)
            for d in range(self.d_architecture.n_dense_layers):
                is_output_layer = (d == self.d_architecture.n_dense_layers - 1)
                x = Dense(self.d_architecture.n_dense_units[d], name='d_h%d_lin'%d)(x)
                if self.d_architecture.use_batch_norm and (self.d_architecture.output_batch_norm or not is_output_layer):
                    x = BatchNormalization()(x)
                if self.d_architecture.use_dropout:
                    x = Dropout(self.d_architecture.dropout_rate)(x)
                if is_output_layer: # output
                    x = Activation('sigmoid')(x)
                else:
                    x = LeakyReLU(alpha=self.cfg.lrelu_alpha)(x)
            
            return Model(image, x, name='D')


    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]

        if  self.cfg.optimizer == 'adam':
            self.optimizer = Adam(lr=self.cfg.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        elif self.cfg.optimizer == 'rmsprop':
            selfoptimizer = RMSprop(lr=self.cfg.learning_rate, clipvalue=1.0, decay=1e-8)

        # Construct discriminator/D network takes real image as input.
        # D - sigmoid and D_logits -linear output.
        self.discriminator = self.build_discriminator(image_dims)

        # Model to train D to discrimate real images.
        self.discriminator.compile(optimizer=self.optimizer, loss='binary_crossentropy')

        # Construct generator/R network.
        self.generator = self.build_generator(image_dims)
        img = Input(shape=image_dims)

        reconstructed_img = self.generator(img)

        self.discriminator.trainable = False
        validity = self.discriminator(reconstructed_img)

        # Model to train Generator/R to minimize reconstruction loss and trick D to see
        # generated images as real ones.
        self.adversarial_model = Model(img, [reconstructed_img, validity])
        self.adversarial_model.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
            loss_weights=[self.r_alpha, 1],
            optimizer=self.optimizer)

        print('\n\rdiscriminator')
        self.discriminator.summary()
        print('\n\rautoencoder')
        self.generator.summary()
        print('\n\radversarial_model')
        self.adversarial_model.summary()


    def train(self, epochs, batch_size = 128, sample_interval=500):
        train_start_time = datetime.datetime.now()
        config_prepend = []
        config_prepend.append("# Training started at: %s"%train_start_time)
        # Make log folder if not exist.
        os.makedirs(self.log_dir, exist_ok=True)
        print('Diagnostics will be save to ', self.log_dir)
        if self.dataset_name in ('mnist','prosivic','dreyeve'):
            # Get a batch of sample images with attention_label to export as montage.
            sample = self.data[0:batch_size]

        # Export images as montage, sample_input also use later to generate sample R network outputs during training.
        sample_inputs = np.array(sample).astype(np.float32)
        os.makedirs(self.train_dir, exist_ok=True)
        scipy.misc.imsave('./{}train_input_samples.jpg'.format(self.train_dir), montage(np.squeeze(sample_inputs)))
        counter = 1
        # Record generator/R network reconstruction training losses.
        self.plot_epochs = []
        self.plot_g_recon_losses = []
        self.plot_g_val_losses = []
        self.plot_d_real_losses = []
        self.plot_d_fake_losses = []

        # Load traning data, add random noise.
        if self.dataset_name in ('mnist','prosivic','dreyeve'):
            sample_w_noise = get_noisy_data(self.data)

        # Adversarial ground truths
        ones = np.ones((batch_size, 1))
        zeros = np.zeros((batch_size, 1))

        checkpoint_interval = max(epochs // self.cfg.num_checkpoints,1)
        epochs_duration = 0
        n_batches = len(self.data) // batch_size
        learning_rate_drop_epoch = int(epochs*self.cfg.learning_rate_drop_epochs_frac)
        for epoch in range(self.start_epoch, epochs):
            epoch_start_time = datetime.datetime.now()
            if self.cfg.print_batch_loss:
                print('Epoch ({}/{})-----------------------------------------------------------------------'.format(epoch+1,epochs))

            if self.cfg.learning_rate_drop and epoch == learning_rate_drop_epoch:
                old_lr = K.eval(self.discriminator.optimizer.lr)
                new_lr = old_lr/self.cfg.learning_rate_drop_factor
                K.set_value(self.discriminator.optimizer.lr, new_lr)
                print("Learning rate change: %f -> %f"%(old_lr, new_lr))

            d_loss_real_epoch = 0
            d_loss_fake_epoch = 0
            g_loss_val_epoch = 0
            g_loss_recon_epoch = 0

            for idx in range(0, n_batches):
                # Get a batch of images and add random noise.
                batch_images = self.data[idx * batch_size:(idx + 1) * batch_size]
                batch_noise_images = sample_w_noise[idx * batch_size:(idx + 1) * batch_size]
                batch_clean_images = self.data[idx * batch_size:(idx + 1) * batch_size]

                # Turn batch images data to float32 type.
                batch_images = np.array(batch_images).astype(np.float32)
                batch_noise_images = np.array(batch_noise_images).astype(np.float32)
                batch_clean_images = np.array(batch_clean_images).astype(np.float32)
                batch_fake_images = self.generator.predict(batch_noise_images)

                # Update D network, minimize real images inputs->D-> ones, noisy z->R->D->zeros loss.
                d_loss_real = self.discriminator.train_on_batch(batch_images, ones)
                d_loss_fake = self.discriminator.train_on_batch(batch_fake_images, zeros)
                # Update R network twice, minimize noisy z->R->D->ones and reconstruction loss.
                self.adversarial_model.train_on_batch(batch_noise_images, [batch_clean_images, ones])
                g_loss = self.adversarial_model.train_on_batch(batch_noise_images, [batch_clean_images, ones])
                g_loss_recon = g_loss[1]
                g_loss_val = g_loss[2]

                # Update arrays for plotting
                d_loss_real_epoch  += d_loss_real
                d_loss_fake_epoch  += d_loss_fake
                g_loss_val_epoch   += g_loss_val
                g_loss_recon_epoch += g_loss_recon

                counter += 1
                if self.cfg.print_batch_loss:
                    msg = 'Epoch:[{0}/{1}]-[{2}/{3}] --> d_loss: {4:>0.6f}, g_val_loss:{5:>0.6f}, g_recon_loss:{6:>0.6f}'.format(epoch+1,epochs, idx+1, n_batches, d_loss_real+d_loss_fake, g_val_loss, g_recon_loss)
                    print(msg)
                    logging.info(msg)

                if np.mod(counter, sample_interval) == 0:
                    samples = self.generator.predict(sample_inputs)
                    scipy.misc.imsave('./{}{:02d}_{:04d}_reconstructions.jpg'.format(self.train_dir,epoch,idx), montage(np.squeeze(samples)))
            
            # end of loop over batches

            # Print epoch loss
            d_loss_real_epoch  = d_loss_real_epoch / n_batches
            d_loss_fake_epoch  = d_loss_fake_epoch / n_batches
            g_loss_val_epoch   = g_loss_val_epoch / n_batches
            g_loss_recon_epoch = g_loss_recon_epoch / n_batches

            msg = 'Epoch:[{0}/{1}] --> d_loss: {2:>0.6f}, g_val_loss:{3:>0.6f}, g_recon_loss:{4:>0.6f}'.format(epoch+1,epochs, d_loss_real_epoch+d_loss_fake_epoch, g_loss_val_epoch, g_loss_recon_epoch)
            print(msg)
            logging.info(msg)

            self.plot_d_real_losses.append(d_loss_real_epoch)
            self.plot_d_fake_losses.append(d_loss_fake_epoch)
            self.plot_g_val_losses.append(g_loss_val_epoch)
            self.plot_g_recon_losses.append(g_loss_recon_epoch)
            self.plot_epochs.append(epoch)

            # Save the checkpoint with specified interval and in last epoch
            if epoch % checkpoint_interval == 0:
                self.save(epoch)
                self.export_loss_plots()

            epoch_end_time = datetime.datetime.now()
            this_epoch_time = (epoch_end_time-epoch_start_time).total_seconds()
            epochs_duration += this_epoch_time
            complete_epochs = epoch+1-self.start_epoch
            ETA = (epochs-complete_epochs)*epochs_duration/complete_epochs
            ETA_str = "%dh%dm%.2fs"%(ETA//3600,(ETA%3600)//60,ETA%60)
            print('Epoch (%d/%d) complete.\tTime: %.2f\tETA: %s'%(epoch+1,epochs,this_epoch_time,ETA_str))
        # end loop over epochs

        s_per_epoch = epochs_duration/epochs

        # Save the last version of the network
        self.save(epochs-1)
        self.export_loss_plots()



        # Save configuration used for the training procedure
        end_time = datetime.datetime.now()
        exp_duration = (end_time-train_start_time).total_seconds()
        config_prepend.append("# Training ended at: %s"%train_start_time)
        config_prepend.append("# Training duration: %dh %dm %.2fs"%(exp_duration//3600,(exp_duration//60)%60, exp_duration%60))
        config_prepend.append("# Training epochs: %d"%epochs)
        config_prepend.append("# Seconds per epoch: %.3f"%s_per_epoch)

        self.save_config(config_prepend)

    def save(self, step):
        """Helper method to save model weights.
        
        Arguments:
            step {[type]} -- [description]
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        adv_model_name = 'ALOCC_Model_{}_adv.h5'.format(step)
        d_model_name = 'ALOCC_Model_{}_d.h5'.format(step)
        self.adversarial_model.save_weights(os.path.join(self.checkpoint_dir, adv_model_name))
        self.discriminator.save_weights(os.path.join(self.checkpoint_dir, d_model_name))

    def load_last_checkpoint(self, epochs=None):
        # Looks in models checkpoint directory and automatically find the latest checkpoint
        # Parameter epochs is the highest epoch a checkpoint will be loaded for
        # If epochs == None (default), the latest checkpoint is found and loaded
        if os.path.exists(self.checkpoint_dir):
            filenames = os.listdir(self.checkpoint_dir)
            filenames = [x.replace('ALOCC_Model_','').replace('.h5','') for x in filenames]
            d_epochs = [int(x.replace('_d','')) for x in filenames if '_d' in x]
            adv_epochs = [int(x.replace('_adv','')) for x in filenames if '_adv' in x]
            try:
                max_epoch = np.intersect1d(d_epochs, adv_epochs)[-1]
                print("Latest common checkpoint found in epoch %d"%max_epoch)
                if epochs is None or max_epoch < epochs-1:
                    d_checkpoint_path = self.checkpoint_dir+'ALOCC_Model_%d_d.h5'%max_epoch
                    adv_checkpoint_path = self.checkpoint_dir+'ALOCC_Model_%d_adv.h5'%max_epoch
                    self.discriminator.load_weights(d_checkpoint_path)
                    self.adversarial_model.load_weights(adv_checkpoint_path)
                    self.start_epoch = max_epoch+1
                    return False
                else: 
                    print("Found checkpoint in later epoch than epochs requested")
                    return True

            except IndexError:
                print("No common checkpoint for discriminator and adversarial model")
                return False
        else:
            print("No checkpoints found")
            return False


    def save_config(self, prepend): 
        """ Save current state of configuration.py for reproduction purposes
        """
        with open(self.log_dir+'configuration.py','w') as outfile:
            with open('./configuration.py','r') as infile:
                for line in prepend:
                    outfile.write(line+'\n')
                for line in infile.readlines():
                    outfile.write(line)

    def export_loss_plots(self):
        # Export the Generator/R network reconstruction losses as a plot.
        plt.clf()
        plt.title('Generator/R network losses')
        plt.xlabel('Epoch')
        plt.ylabel('training loss')
        plt.grid()
        plt.plot(self.plot_epochs,self.plot_g_recon_losses, label="Reconstruction loss")

        # Export the Generator/R network validity losses as a plot.
        plt.xlabel('Epoch')
        plt.ylabel('training loss')
        plt.grid()
        plt.plot(self.plot_epochs,self.plot_g_val_losses, label="Validity loss")
        plt.legend()
        plt.savefig(self.train_dir+'plot_g_losses.png')

        plt.clf()
        # Export the discriminator losses for real images as a plot.
        plt.title('Discriminator loss for real/fake images')
        plt.xlabel('Epoch')
        plt.ylabel('training loss')
        plt.grid()
        plt.plot(self.plot_epochs,self.plot_d_real_losses, label="Real images")

        # Export the discriminator losses for fake images as a plot.
        plt.xlabel('Epoch')
        plt.ylabel('training loss')
        plt.grid()
        plt.plot(self.plot_epochs,self.plot_d_fake_losses, label="Generator images")
        plt.legend()
        plt.savefig(self.train_dir+'plot_d_losses.png')
    
    
if __name__ == '__main__':
    parser=argparse.ArgumentParser()


    parser.add_argument('--epochs', '-e', type=int, default=None, help='Epochs to train for (overrides configuration.py')
    parser.add_argument('--exp_name', '-x', default='default', help='Unique name of experiment (overrides configuration.py)')
    parser.add_argument('--batch_size', '-b', type=int, default=None, help='Size of minibatches during training (overrides configuration.py)')
    parser.add_argument('--dataset', '-d', default='mnist', help='Dataset to use for experiment (overrides configuration.py)')
    
    args=parser.parse_args()
    epochs = args.epochs
    exp_name = args.exp_name
    batch_size = args.batch_size
    dataset = args.dataset

    cfg = Configuration(dataset, exp_name)

    if epochs is None:
        epochs = cfg.n_epochs
    if batch_size is None:
        batch_size = cfg.batch_size

    log_dir = './log/'+dataset+'/'+exp_name+'/'

    print("Dataset: ", dataset)
    print("Training for %d epochs"%epochs)

    model = ALOCC_Model(dataset_name=dataset, input_height=cfg.image_height,input_width=cfg.image_width, r_alpha = cfg.r_alpha, log_dir = log_dir, experiment_name=exp_name, cfg = cfg)
    training_complete = model.load_last_checkpoint(epochs)
    
    if not training_complete:
        model.train(epochs=epochs, batch_size=batch_size, sample_interval=min([500,cfg.n_train]))
