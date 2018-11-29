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
from configuration import Configuration as cfg
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
               z_dim=100, gf_dim=16, df_dim=16, c_dim=3,
               dataset_name=None, dataset_address=None, input_fname_pattern=None,
               log_dir=cfg.log_dir, r_alpha = 0.2,
               kb_work_on_patch=True, nd_patch_size=(10, 10), n_stride=1,
               n_fetch_data=10, outlier_dir = cfg.test_out_folder):
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

        self.b_work_on_patch = kb_work_on_patch

        # Create different log dirs
        self.log_dir = log_dir
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

        if cfg.hardcoded_architecture == 'ALOCC_mnist':
            print("Using original ALOCC architectures")
            self.ae_architecture = None
            self.d_architecture = None
        else:
            self.ae_architecture = AE_Architecture(hardcoded = cfg.hardcoded_architecture)
            self.d_architecture = D_Architecture(hardcoded = cfg.hardcoded_architecture)
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
               X_test_in = inlier_data[np.random.choice(len(inlier_data), cfg.n_test_in, replace=False)]
               n_test_out = cfg.n_test - cfg.n_test_in
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
                X_train = np.array([img_to_array(load_img(cfg.train_folder + filename)) for filename in os.listdir(cfg.train_folder)][:cfg.n_train])
                self.data = X_train / 255.0
                # self._X_val = [img_to_array(load_img(Cfg.prosivic_val_folder + filename)) for filename in os.listdir(Cfg.prosivic_val_folder)][:Cfg.prosivic_n_val] 
            else: #load test data     
                n_test_out = cfg.n_test - cfg.n_test_in
                X_test_in = np.array([img_to_array(load_img(cfg.test_in_folder + filename)) for filename in os.listdir(cfg.test_in_folder)][:cfg.n_test_in])
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
            x = LeakyReLU()(x)
            x = Conv2D(filters=self.df_dim * 4, kernel_size = 5, strides=2, padding='same', name='g_encoder_h1_conv')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = Conv2D(filters=self.df_dim * 8, kernel_size = 5, strides=2, padding='same', name='g_encoder_h2_conv')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)

            # Decoder.
            # TODO: need a flexable solution to select output_padding and padding.
            x = Conv2DTranspose(self.gf_dim*2, kernel_size = 5, strides=2, activation='relu', padding='same', output_padding=0, name='g_decoder_h0')(x)
            x = BatchNormalization()(x)
            x = Conv2DTranspose(self.gf_dim*1, kernel_size = 5, strides=2, activation='relu', padding='same', output_padding=1, name='g_decoder_h1')(x)
            x = BatchNormalization()(x)
            x = Conv2DTranspose(self.c_dim,    kernel_size = 5, strides=2, activation='tanh', padding='same', output_padding=1, name='g_decoder_h2')(x)

            #x = Conv2D(self.gf_dim*1, kernel_size=5, activation='relu', padding='same')(x)
            #x = UpSampling2D((2, 2))(x)
            #x = Conv2D(self.gf_dim*1, kernel_size=5, activation='relu', padding='same')(x)
            #x = UpSampling2D((2, 2))(x)
            #x = Conv2D(self.gf_dim*2, kernel_size=3, activation='relu')(x)
            #x = UpSampling2D((2, 2))(x)
            #x = Conv2D(self.c_dim, kernel_size=5, activation='sigmoid', padding='same')(x)
            return Model(image, x, name='R')
        
        else: # architecture built from ae_architecture object
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
                    x = LeakyReLU()(x)
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
                    x = LeakyReLU()(x)

            # Decoder
            if self.ae_architecture.n_dense_layers > 0:
                n_dense_units_flip = np.flip(self.ae_architecture.n_dense_units)[:-1]
                n_dense_units_flip = np.append(n_dense_units_flip,np.prod(shape_before_dense))

                for d in range(self.ae_architecture.n_dense_layers):
                    x = Dense(n_dense_units_flip[d], name='g_decoder_h%d_lin'%d)(x)
                    if self.ae_architecture.use_batch_norm:
                        x = BatchNormalization()(x)
                    if self.ae_architecture.use_dropout:
                        x = Dropout(self.ae_architecture.dropout_rate)(x)
                    x = LeakyReLU()(x)

                x = Reshape(shape_before_dense)(x)

            n_conv_layers_per_module_flip = np.flip(self.ae_architecture.n_conv_layers_per_module)
            channels_flip = np.flip(self.ae_architecture.channels)[:-1]
            channels_flip = np.append(channels_flip, self.c_dim)
            filter_size_flip = np.flip(self.ae_architecture.filter_size)
            stride_flip = np.flip(self.ae_architecture.stride)

            for m in range(self.ae_architecture.n_conv_modules): # Loop over conv modules
                channels = channels_flip[m]
                if self.ae_architecture.max_pool:
                    x = UpSampling2D((self.ae_architecture.pool_size,self.ae_architecture.pool_size))(x)
                for l in range(n_conv_layers_per_module_flip[m]):
                    stride = int(stride_flip[m])
                    k_size = int(filter_size_flip[m])
                    if l == 0: # last layer in module
                        stride = self.ae_architecture.dim_red_stride[m] # sets stride to 2 in last layer if max_pool = False
                    if self.ae_architecture.max_pool:
                        x = Conv2DTranspose(filters=channels, kernel_size = k_size, strides=stride, padding='same',  name='g_decoder_h%d_%d_conv'%(m,l))(x)
                    else:
                        outpad = int((k_size-stride)%2)
                        inpad = int((k_size-stride+outpad)//2)
                        #print(inpad, type(inpad))
                        #x = ZeroPadding2D(padding=inpad)(x)
#                        x = Conv2DTranspose(filters=channels, kernel_size = k_size, strides=stride, padding='valid', output_padding = (outpad,outpad), name='g_decoder_h%d_%d_conv'%(m,l))(x)
                        x = Conv2DTranspose(filters=channels, kernel_size = k_size, strides=stride, padding='same', name='g_decoder_h%d_%d_conv'%(m,l))(x)
                        if self.ae_architecture.use_batch_norm:
                            x = BatchNormalization()(x)
                    if self.ae_architecture.use_dropout:
                        x = Dropout(self.ae_architecture.dropout_rate)(x)
                    if m == self.ae_architecture.n_conv_modules - 1 and l == self.ae_architecture.n_conv_layers_per_module[m]-1:
                        # Output layer
                        x = Activation('sigmoid')(x)
                    else:
                        x = LeakyReLU()(x)
            
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
            x = LeakyReLU()(x)

            x = Conv2D(filters=self.df_dim*2, kernel_size = 5, strides=2, padding='same', name='d_h1_conv')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)

            x = Conv2D(filters=self.df_dim*4, kernel_size = 5, strides=2, padding='same', name='d_h2_conv')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)

            x = Conv2D(filters=self.df_dim*8, kernel_size = 5, strides=2, padding='same', name='d_h3_conv')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)

            x = Flatten()(x)
            x = Dense(1, activation='sigmoid', name='d_h3_lin')(x)

            return Model(image, x, name='D')
        
        else: # architecture built from d_architecture object
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
                    x = LeakyReLU()(x)
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
                x = Dense(self.d_architecture.n_dense_units[d], name='d_h%d_lin'%d)(x)
                if self.d_architecture.use_batch_norm:
                    x = BatchNormalization()(x)
                if self.d_architecture.use_dropout:
                    x = Dropout(self.d_architecture.dropout_rate)(x)
                if d == self.d_architecture.n_dense_layers - 1: # output
                    x = Activation('sigmoid')(x)
                else:
                    x = LeakyReLU()(x)
            
            return Model(image, x, name='D')


    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]
        optimizer = RMSprop(lr=0.002, clipvalue=1.0, decay=1e-8)
        # Construct discriminator/D network takes real image as input.
        # D - sigmoid and D_logits -linear output.
        self.discriminator = self.build_discriminator(image_dims)

        # Model to train D to discrimate real images.
        self.discriminator.compile(optimizer=optimizer, loss='binary_crossentropy')

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
            optimizer=optimizer)

        print('\n\rdiscriminator')
        self.discriminator.summary()
        print('\n\rautoencoder')
        self.generator.summary()
        print('\n\radversarial_model')
        self.adversarial_model.summary()


    def train(self, epochs, batch_size = 128, sample_interval=500):
        start_time = datetime.datetime.now
        config_prepend = []
        config_prepend.append("Training started at: %s"%start_time)
        # Make log folder if not exist.
        os.makedirs(self.log_dir, exist_ok=True)
        print('Diagnostics will be save to ', self.log_dir)
        if self.dataset_name in ('mnist','prosivic','dreyeve'):
            # Get a batch of sample images with attention_label to export as montage.
            sample = self.data[0:batch_size]

        # Export images as montage, sample_input also use later to generate sample R network outputs during training.
        sample_inputs = np.array(sample).astype(np.float32)
        os.makedirs(self.train_dir, exist_ok=True)
        scipy.misc.imsave('./{}train_input_samples.jpg'.format(self.train_dir), montage(sample_inputs[:,:,:,0]))

        counter = 1
        # Record generator/R network reconstruction training losses.
        plot_epochs = []
        plot_g_recon_losses = []
        plot_d_real_losses = []
        plot_d_fake_losses = []

        # Load traning data, add random noise.
        if self.dataset_name in ('mnist','prosivic','dreyeve'):
            sample_w_noise = get_noisy_data(self.data)

        # Adversarial ground truths
        ones = np.ones((batch_size, 1))
        zeros = np.zeros((batch_size, 1))

        checkpoint_interval = epochs // cfg.num_checkpoints

        for epoch in range(epochs):
            print('Epoch ({}/{})-------------------------------------------------'.format(epoch,epochs))
            if self.dataset_name in ('mnist','prosivic','dreyeve'):
                # Number of batches computed by total number of target data / batch size.
                batch_idxs = len(self.data) // batch_size
             
            for idx in range(0, batch_idxs):
                # Get a batch of images and add random noise.
                if self.dataset_name in ('mnist','prosivic','dreyeve'):
                    batch = self.data[idx * batch_size:(idx + 1) * batch_size]
                    batch_noise = sample_w_noise[idx * batch_size:(idx + 1) * batch_size]
                    batch_clean = self.data[idx * batch_size:(idx + 1) * batch_size]
                # Turn batch images data to float32 type.
                batch_images = np.array(batch).astype(np.float32)
                batch_noise_images = np.array(batch_noise).astype(np.float32)
                batch_clean_images = np.array(batch_clean).astype(np.float32)
                if self.dataset_name in ('mnist','prosivic','dreyeve'):
                    batch_fake_images = self.generator.predict(batch_noise_images)
                    # Update D network, minimize real images inputs->D-> ones, noisy z->R->D->zeros loss.
                    d_loss_real = self.discriminator.train_on_batch(batch_images, ones)
                    d_loss_fake = self.discriminator.train_on_batch(batch_fake_images, zeros)
                    plot_d_real_losses.append(d_loss_real)
                    plot_d_fake_losses.append(d_loss_fake)

                    # Update R network twice, minimize noisy z->R->D->ones and reconstruction loss.
                    self.adversarial_model.train_on_batch(batch_noise_images, [batch_clean_images, ones])
                    g_loss = self.adversarial_model.train_on_batch(batch_noise_images, [batch_clean_images, ones])    
                    plot_epochs.append(epoch+idx/batch_idxs)
                    plot_g_recon_losses.append(g_loss[1])
                counter += 1
                msg = 'Epoch:[{0}]-[{1}/{2}] --> d_loss: {3:>0.3f}, g_loss:{4:>0.3f}, g_recon_loss:{4:>0.3f}'.format(epoch+1, idx+1, batch_idxs, d_loss_real+d_loss_fake, g_loss[0], g_loss[1])
                print(msg)
                logging.info(msg)
                if np.mod(counter, sample_interval) == 0:
                    if self.dataset_name == 'mnist':
                        samples = self.generator.predict(sample_inputs)
                        #manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                        #manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
                        #save_images(samples, [manifold_h, manifold_w],
                        #    './{}/train_{:02d}_{:04d}.png'.format(self.train_dir, epoch, idx))
                        #scipy.misc.imsave(self.train_dir+'train_%d_%d_samples.png'%(epoch,idx), montage(np.squeeze(samples)))
                        scipy.misc.imsave('./{}{:02d}_{:04d}_reconstructions.jpg'.format(self.train_dir,epoch,idx), montage(np.squeeze(samples)))

            # Save the checkpoint end of each epoch.
            if epoch % checkpoint_interval == 0:
                self.save(epoch)

        # Save the last version of the network
        self.save("final")

        # Export the Generator/R network reconstruction losses as a plot.
        plt.title('Generator/R network reconstruction losses')
        plt.xlabel('Epoch')
        plt.ylabel('training loss')
        plt.grid()
        plt.plot(plot_epochs,plot_g_recon_losses)
        plt.savefig(self.train_dir+'plot_g_recon_losses.png')

        plt.clf()
        # Export the discriminator losses for real images as a plot.
        plt.title('Discriminator loss for real images')
        plt.xlabel('Epoch')
        plt.ylabel('training loss')
        plt.grid()
        plt.plot(plot_epochs,plot_d_real_losses)
        plt.savefig(self.train_dir+'plot_d_real_losses.png')

        plt.clf()
        # Export the discriminator losses for fake images as a plot.
        plt.title('Discriminator loss for fake images')
        plt.xlabel('Epoch')
        plt.ylabel('training loss')
        plt.grid()
        plt.plot(plot_epochs,plot_d_fake_losses)
        plt.savefig(self.train_dir+'plot_d_fake_losses.png')

        # Save configuration used for the training procedure
        end_time = datetime.datetime.now()
        exp_duration = (end_time-start_time).total_seconds()
        config_prepend.append("Training ended: %s"%end_time.strftime("%A, %d. %B %Y %I:%M%p"))
        config.prepend.append("Training duration: %dh %dm %.2fs"%(exp_duration//3600,(exp_duration//60)%60, exp_duration%60))
        self.save_config()

    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.dataset_name,
            self.output_height, self.output_width)

    def save(self, step):
        """Helper method to save model weights.
        
        Arguments:
            step {[type]} -- [description]
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        model_name = 'ALOCC_Model_{}.h5'.format(step)
        self.adversarial_model.save_weights(os.path.join(self.checkpoint_dir, model_name))

    def save_config(self, prepend): 
        """ Save current state of configuration.py for reproduction purposes
        """
        with open(self.log_dir+'configuration.py','w') as outfile:
            with open('./configuration.py','r') as infile:
                for line in prepend:
                    outfile.write(line+'\n')
                for line in infile.readlines():
                    outfile.write(line)
        
if __name__ == '__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument('--epochs', '-e', type=int, default=cfg.n_epochs, help='Epochs to train for (overrides configuration.py')
    parser.add_argument('--exp_name', '-x', default=cfg.experiment_name, help='Unique name of experiment (overrides configuration.py)')
    parser.add_argument('--batch_size', '-b', type=int, default=cfg.batch_size, help='Size of minibatches during training (overrides configuration.py)')
    args=parser.parse_args()
    epochs = args.epochs
    exp_name = args.exp_name
    batch_size = args.batch_size

    dataset = cfg.dataset

    log_dir = './log/'+dataset+'/'+exp_name+'/'

    print("Dataset: ", dataset)
    print("Training for %d epochs"%epochs)

    model = ALOCC_Model(dataset_name=dataset, input_height=cfg.image_height,input_width=cfg.image_width, r_alpha = cfg.r_alpha, log_dir = log_dir)
    model.train(epochs=epochs, batch_size=batch_size, sample_interval=min([500,cfg.n_train]))
