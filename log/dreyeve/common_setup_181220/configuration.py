# Training started at: 2018-12-21 01:28:01.996823
# Training ended at: 2018-12-21 01:28:01.996823
# Training duration: 12h 25m 29.38s
# Training epochs: 500
# Seconds per epoch: 89.326
from pathlib import Path
import datetime
class Configuration(object):

    def __init__(self, dataset = 'mnist', experiment_name = 'default'):

        self.dataset = dataset

        # General trianing settings
        self.r_alpha = 0.2
        self.learning_rate = 0.001
        self.learning_rate_drop = True
        self.learning_rate_drop_epochs_frac = 0.5
        self.learning_rate_drop_factor = 10
        self.optimizer = 'adam'
        self.decay = 0.0
        self.lrelu_alpha = 0.01
        self.alternate_training = False
        self.alternate_epochs = 5
        # Log settings
        self.experiment_name = experiment_name

        # Diagnostics settings
        self.print_batch_loss = False
        # Test settings
        self.load_epoch = "final"
        self.test_batch_size = 64
        self.test_batch_verbose = False

        # Dataset specific settings below
        if self.dataset == 'mnist':

            self.image_height = 28
            self.image_width = 28
            self.channels = 1
            self.batch_size = 128
            self.hardcoded_architecture = 'ALOCC_mnist'
            self.n_epochs = 10
            self.n_test = 5000
            self.n_test_in = 2500
            self.num_checkpoints = 10
            self.n_train = 5000
            self.test_out_folder = ''

        if self.dataset == "dreyeve":
            # Autoencoder architecture
            self.hardcoded_architecture = None
            self.ae_n_conv_modules = 6 # number of conv. modules
            self.ae_n_conv_layers_per_module = 1 # number of conv. layers in each module (between each pool layer/dim reduction)
            self.ae_n_dense_layers = 1 # number of dense layers in
            self.ae_z_dim = 512
            self.ae_n_dense_units = self.ae_z_dim
            self.ae_filter_size = 5
            self.ae_stride = 1 # will be set to ae_pool_size if ae_max_pool is False
            self.ae_max_pool = False
            self.ae_pool_size = 2
            self.ae_first_layer_channels = 16
            self.ae_channels = [16 * 2**i for i in range(self.ae_n_conv_modules)]
            self.ae_use_batch_norm = True
            self.ae_output_batch_norm = True
            self.ae_use_dropout = False
            self.ae_dropout_rate = 0.1

            # Discriminator architecture
            self.d_n_conv_modules = 7 # number of conv. modules
            self.d_n_conv_layers_per_module = 1 # number of conv. layers in each module (between each pool layer/dim reduction)
            self.d_n_dense_layers = 1 # number of dense layers in
            self.d_z_dim = 1
            self.d_n_dense_units = 1
            self.d_filter_size = 5
            self.d_stride = 1
            self.d_max_pool = False
            self.d_pool_size = 2
            self.d_channels_first_layer = 64
            self.d_channels = [16 * 2**i for i in range(self.d_n_conv_modules)]
            self.d_use_batch_norm = True
            self.d_output_batch_norm = True
            self.d_use_dropout = False
            self.d_dropout_rate = 0.1

            # Data format
            self.image_height = 256
            self.image_width = 256
            self.channels = 3

            # Train settings
            self.data_divider = 1
            self.n_epochs = 500
            self.n_train = 6000//self.data_divider
            self.n_val = 600//self.data_divider
            self.n_test = 1200//self.data_divider
            self.n_test_in = 600 // self.data_divider
            self.out_frac = (self.n_test-self.n_test_in)/self.n_test
            self.batch_size = 64
            self.num_checkpoints = 10

            # Data sources
            self.img_folder =   "../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/"
            self.train_folder =   "../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/train/"
            self.val_folder =   "../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/val/"
            self.test_folder = "../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/test/"
            self.test_in_folder =  self.test_folder + "in/"
            self.test_out_folder =  self.test_folder + "out/"

        if self.dataset == "prosivic":
            self.hardcoded_architecture = None
            # Autoencoder architecture
            self.ae_n_conv_modules =  5 # number of conv. modules
            self.ae_n_conv_layers_per_module = 1 # number of conv. layers in each module (between each pool layer/dim reduction)
            self.ae_n_dense_layers = 1 # number of dense layers in
            self.ae_z_dim = 512
            self.ae_n_dense_units = self.ae_z_dim
            self.ae_filter_size = 5
            self.ae_stride = 1 # will be set to ae_pool_size automatically if ae_max_pool is False
            self.ae_max_pool = False
            self.ae_pool_size = 2
            self.ae_first_layer_channels = 16
            self.ae_channels = [16 * 2**i for i in range(self.ae_n_conv_modules)] # TODO: replace number with ae_first_layer_channels
            self.ae_use_batch_norm = True
            self.ae_output_batch_norm = True
            self.ae_use_dropout = False
            self.ae_dropout_rate = 0.1

            # Discriminator architecture
            self.d_n_conv_modules =  6 # number of conv. modules
            self.d_n_conv_layers_per_module = 1 # number of conv. layers in each module (between each pool layer/dim reduction)
            self.d_n_dense_layers = 1 # number of dense layers in
            self.d_z_dim = 512
            self.d_n_dense_units = 1
            self.d_filter_size = 5
            self.d_stride = 1
            self.d_max_pool = False
            self.d_pool_size = 2
            self.d_channels_first_layer = 16
            self.d_channels = [16 * 2**i for i in range(self.d_n_conv_modules)] # TODO: replace number with d_channels_first_layer
            self.d_use_batch_norm = True
            self.d_output_batch_norm = True
            self.d_use_dropout = False
            self.d_dropout_rate = 0.1

            # Train settings
            self.n_epochs = 500
            self.data_div = 1
            self.n_train = 6785 // self.data_div
            self.n_val = 840 // self.data_div
            self.n_test = 1000 // self.data_div
            self.n_test_in = 500 // self.data_div
            self.out_frac = (self.n_test-self.n_test_in)/self.n_test
            self.batch_size = 64
            self.num_checkpoints  = 10

            # Data format
            self.image_height = 256
            self.image_width = 256
            self.channels = 3

            # Data sources
            self.img_folder =   "../weather_detection_data/prosivic/"
            self.train_folder = "../weather_detection_data/prosivic/train/"
            self.val_folder =   "../weather_detection_data/prosivic/val/"
            self.test_folder = "../weather_detection_data/prosivic/test/"
            self.test_in_folder =  self.test_folder + "in/"
            self.test_out_folder =  self.test_folder + "out/foggy/"

        '''
        elif self.dataset == "bdd100k":
            self.img_folder = Path("/data/bdd100k/images/train_and_val_256by256")
            self.norm_file = "/data/bdd100k/namelists/clear_or_partly_cloudy_or_overcast_and_highway_and_daytime.txt"
            self.norm_filenames = loadbdd100k.get_namelist_from_file(norm_file)
            self.out_file = "/data/bdd100k/namelists/rainy_or_snowy_or_foggy_and_highway_and_daytime_or_dawndusk_or_night.txt"
            self.out_filenames = loadbdd100k.get_namelist_from_file(out_file)
            self.norm_spec = [["weather", ["clear","partly cloudy", "overcast"]],["scene", "highway"],["timeofday", "daytime"]]
            self.out_spec = [["weather", ["rainy", "snowy", "foggy"]],["scene", "highway"],["timeofday",["daytime","dawn/dusk","night"]]]
            self.save_name_lists=False
            self.labels_file = None
            self.get_norm_and_out_sets = False
            self.shuffle=False
            self.architecture = "b1"
        '''
        

        # generated config settings
        self.log_dir = './log/'+self.dataset+'/'+self.experiment_name+'/'
        self.model_dir = self.log_dir+'models/'
        self.train_dir = self.log_dir+'train/'
        self.test_dir = self.log_dir+'test/'
        self.test_batch_size = min(self.test_batch_size, self.n_test)
################################################################
# Test started at: 2019-01-06 17:51:05.745978
#Num outliers: 600
#Num inliers: 600
# Dataset: dreyeve, outliers: out/
# AUROC D(R(x))-score:	0.49833
# AUPRC D(R(x))-score:	0.74791
# AUROC rec_err:	0.56035
# AUPRC: rec_err	0.51924
################################################################
# Test started at: 2019-01-06 17:52:31.255503
#Num outliers: 600
#Num inliers: 600
# Dataset: dreyeve, outliers: out/
# AUROC D(R(x))-score:	0.50700
# AUPRC D(R(x))-score:	0.74101
# AUROC rec_err:	0.70544
# AUPRC: rec_err	0.64070
################################################################
# Test started at: 2019-01-06 17:55:30.199577
#Num outliers: 600
#Num inliers: 600
# Dataset: dreyeve, outliers: out/
# AUROC D(R(x))-score:	0.50700
# AUPRC D(R(x))-score:	0.74101
# AUROC rec_err:	0.70544
# AUPRC: rec_err	0.64070
################################################################
# Test started at: 2019-01-06 17:57:12.912113
#Num outliers: 600
#Num inliers: 600
# Dataset: dreyeve, outliers: out/
# AUROC D(R(x))-score:	0.49833
# AUPRC D(R(x))-score:	0.74791
# AUROC rec_err:	0.56035
# AUPRC: rec_err	0.51924
