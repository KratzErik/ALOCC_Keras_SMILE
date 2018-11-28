from pathlib import Path
import datetime
class Configuration(object):

    dataset = 'prosivic' # will affect which dataset specific settings are used

    # Model hyper parameters
    r_alpha = 0.2

    # Log settings
    experiment_name = 'debug'
    #time_stamp = datetime.datetime.now().strftime("%y_%m_%d_kl%H_%M")
    
    # Test settings
    load_epoch = 90
    test_batch_size = 64
    test_batch_verbose = False

    # Dataset specific settings below
    if dataset == 'mnist':

        image_height = 28
        image_width = 28
        channels = 1
        batch_size = 128
        hardcoded_architecture = 'ALOCC_mnist'
        n_epochs = 10
        n_test = 5000
        n_test_in = 2500
        checkpoint_interval = 2
        n_train = 5000

    if dataset == "dreyeve":
        # Autoencoder architecture
        hardcoded_architecture = None
        ae_n_conv_modules =  4 # number of conv. modules
        ae_n_conv_layers_per_module = 1 # number of conv. layers in each module (between each pool layer/dim reduction)
        ae_n_dense_layers = 1 # number of dense layers in
        ae_z_dim = 256
        ae_n_dense_units = ae_z_dim
        ae_filter_size = 4
        ae_stride = 2
        ae_max_pool = False
        ae_pool_size = 2
        ae_first_layer_channels = 32
        ae_channels = [ae_first_layer_channels * 2**i for i in range(ae_n_conv_modules)]
        ae_use_batch_norm = True
        ae_use_dropout = False
        ae_dropout_rate = 0.1

        # Discriminator architecture
        d_n_conv_modules =  4 # number of conv. modules
        d_n_conv_layers_per_module = 1 # number of conv. layers in each module (between each pool layer/dim reduction)
        d_n_dense_layers = 1 # number of dense layers in
        d_z_dim = 256
        d_n_dense_units = 1
        d_filter_size = 4
        d_stride = 2
        d_max_pool = False
        d_pool_size = 2
        d_channels_first_layer = 16
        d_channels = [d_channels_first_layer * 2**i for i in range(d_n_conv_modules)]
        d_use_batch_norm = True
        d_use_dropout = False
        d_dropout_rate = 0.1

        # Data format
        image_height = 256
        image_width = 256
        channels = 3

        # Train settings
        n_epochs = 40
        n_train = 100
        n_val = 50
        n_test = 100
        n_test_in = 50
        out_frac = (n_test-n_test_in)/n_test
        batch_size = 64
        checkpoint_interval = n_epochs//10

        # Data sources
        img_folder =   "../weather_detection_data/dreyeve/highway_morning_sunny_vs_rainy/"
        train_folder = "../weather_detection_data/dreyeve/highway_morning_sunny_vs_rainy/train/"
        val_folder =   "../weather_detection_data/dreyeve/highway_morning_sunny_vs_rainy/val/"
        test_in_folder =  "../weather_detection_data/dreyeve/highway_morning_sunny_vs_rainy/test/out/"
        test_out_folder =  "../weather_detection_data/dreyeve/highway_morning_sunny_vs_rainy/test/out/"
    
    elif dataset == "prosivic":
        hardcoded_architecture = None
        # Autoencoder architecture
        ae_n_conv_modules =  5 # number of conv. modules
        ae_n_conv_layers_per_module = 1 # number of conv. layers in each module (between each pool layer/dim reduction)
        ae_n_dense_layers = 0 # number of dense layers in
        ae_z_dim = 256
        ae_n_dense_units = ae_z_dim
        ae_filter_size = 4
        ae_stride = 1
        ae_max_pool = False
        ae_pool_size = 2
        ae_first_layer_channels = 16
        ae_channels = [16 * 2**i for i in range(ae_n_conv_modules)] # TODO: replace number with ae_first_layer_channels
        ae_use_batch_norm = True
        ae_use_dropout = False
        ae_dropout_rate = 0.1

        # Discriminator architecture
        d_n_conv_modules =  4 # number of conv. modules
        d_n_conv_layers_per_module = 1 # number of conv. layers in each module (between each pool layer/dim reduction)
        d_n_dense_layers = 1 # number of dense layers in
        d_z_dim = 256
        d_n_dense_units = 1
        d_filter_size = 4
        d_stride = 1
        d_max_pool = False
        d_pool_size = 2
        d_channels_first_layer = 16
        d_channels = [8 * 2**i for i in range(d_n_conv_modules)] # TODO: replace number with d_channels_first_layer
        d_use_batch_norm = True
        d_use_dropout = False
        d_dropout_rate = 0.1

        # Train settings
        n_epochs = 100
        n_train = 100
        n_val = 50
        n_test = 1000
        n_test_in = 500
        out_frac = (n_test-n_test_in)/n_test
        batch_size = 64
        checkpoint_interval = n_epochs//10

        # Data format
        image_height = 256
        image_width = 256
        channels = 3
        batch_size = 64

        # Data sources
        img_folder =   "../weather_detection_data/prosivic/"
        train_folder = "../weather_detection_data/prosivic/train/"
        val_folder =   "../weather_detection_data/prosivic/val/"
        test_in_folder =  "../weather_detection_data/prosivic/test/in/"
        test_out_folder =  "../weather_detection_data/prosivic/test/out/urban/"


    elif dataset == "bdd100k":
        img_folder = Path("/data/bdd100k/images/train_and_val_256by256")
        norm_file = "/data/bdd100k/namelists/clear_or_partly_cloudy_or_overcast_and_highway_and_daytime.txt"
        norm_filenames = loadbdd100k.get_namelist_from_file(norm_file)
        out_file = "/data/bdd100k/namelists/rainy_or_snowy_or_foggy_and_highway_and_daytime_or_dawndusk_or_night.txt"
        out_filenames = loadbdd100k.get_namelist_from_file(out_file)
        norm_spec = [["weather", ["clear","partly cloudy", "overcast"]],["scene", "highway"],["timeofday", "daytime"]]
        out_spec = [["weather", ["rainy", "snowy", "foggy"]],["scene", "highway"],["timeofday",["daytime","dawn/dusk","night"]]]
        save_name_lists=False
        labels_file = None
        get_norm_and_out_sets = False
        shuffle=False
        architecture = "b1"
    

    # generated config settings
    log_dir = './log/'+dataset+'/'+experiment_name+'/'
    model_dir = log_dir+'models/'
    train_dir = log_dir+'train/'
    test_dir = log_dir+'test/'
