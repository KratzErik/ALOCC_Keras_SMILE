class AE_Architecture():
        def __init__(self, cfg = None):
            if cfg.hardcoded is None: # editable configuration
                # Layers
                self.n_conv_modules = cfg.ae_n_conv_modules  # number of conv. modules
                self.n_conv_layers_per_module = cfg.ae_n_conv_layers_per_module # number of conv. layers in each module (between each pool layer/dim reduction)
                self.n_dense_layers = cfg.ae_n_dense_layers # number of dense layers in 
                self.n_dense_units = cfg.ae_z_dim
                if isinstance(self.n_conv_layers_per_module,int):
                    self.n_conv_layers_per_module = [self.n_conv_layers_per_module]*self.n_conv_modules
                if isinstance(self.n_dense_units,int):
                    self.n_dense_units = [self.n_dense_units]*self.n_dense_layers

                # Filters
                self.filter_size = cfg.ae_filter_size
                self.stride = cfg.ae_stride
                self.channels = cfg.ae_channels # num output channels/filters in each conv. module

                if isinstance(self.filter_size,int):
                    self.filter_size = [self.filter_size] * (self.n_conv_modules)
                if isinstance(self.stride,int):
                    self.stride = [self.stride] * (self.n_conv_modules)
                if isinstance(self.channels,int):
                    self.channels = [self.channels] * self.n_conv_modules

                # Other layers
                self.max_pool = cfg.ae_max_pool
                self.pool_size = cfg.ae_pool_size
                if isinstance(self.pool_size,int):
                    self.pool_size = [self.pool_size]*self.n_conv_modules
                self.use_batch_norm = cfg.ae_use_batch_norm
                self.use_dropout = cfg.ae_use_dropout
                self.dropout_rate = cfg.ae_dropout_rate

                if self.max_pool:
                    self.dim_red_stride = self.stride
                else:
                    self.dim_red_stride = self.pool_size

            if cfg.hardcoded == 'VGG16':
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
