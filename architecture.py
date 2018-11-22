from models import ALOCC_Model
from configuration import Configuration as cfg

class Architecture:
        def __init__(self, name = None, model = ALOCC_Model()):
            if name is None:
                # Layers
                self.n_conv_modules =  # number of conv. modules
                self.n_conv_layers_per_module = 1 # number of conv. layers in each module (between each pool layer/dim reduction)
                self.n_dense_layers = 2 # number of dense layers in 
                self.n_dense_units = model.z_dim
                if len(self.n_conv_layers_per_module) == 1:
                    self.n_conv_layers_per_module = [self.n_conv_layers_per_module]*self.n_conv_modules
                if len(self.n_dense_units) == 1:
                    self.n_dense_units = [self.n_dense_units]*self.n_dense_layers

                # Filters
                self.filter_size = 4
                self.stride = 2
                self.channels = [8,16,32,64,128] # num output channels/filters in each conv. module

                if len(self.channel_factor) == 1:
                    self.channel_factor = [self.channel_factor] * (self.n_conv_modules-1)

                # Other layers
                self.max_pool = cfg.max_pool
                self.pool_size = 2
                if len(self.pool_size) == 1:
                    self.pool_size = [self.pool_size]*self.n_conv_modules
                self.use_batch_norm = cfg.use_batch_norm
                self.use_dropout = cfg.use_dropout
                self.dropout_rate = cfg.dropout_rate

                if self.max_pool:
                    self.dim_red_stride = [self.stride]*self.n_conv_modules
                else:
                    self.dim_red_stride = [self.pool_size]

            if name == 'VGG16':
                self.n_conv_modules = 5 # number of conv. modules
                self.n_conv_layers_per_module = [2,2,3,3,3] # number of conv. layers in each module (between each pool layer/dim reduction)
                self.n_dense_layers = 3 # number of dense layers in 
                self.n_dense_units = [4096, 4096, 1000]
                
                # Filters
                self.filter_size = 4
                self.stride = 2
                self.channel_factor = [2,2,2,1] # increase/decrease factor of num channels after each conv. module. Scalar or list with n_conv_modules-1 elements.
                self.init_channels = 64 # num channels after in first conv. module (closest to input/reconstruction layer)
                
                # Other layers
                self.max_pool = False
                self.use_batch_norm = True
                self.use_dropout = False
