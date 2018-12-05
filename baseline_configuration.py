from pathlib import Path
import datetime
class Configuration(object):

    def __init__(self, dataset = 'mnist', experiment_name = 'default'):

        self.dataset = dataset

        # Log settings
        self.experiment_name = experiment_name

        # Test settings

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

            # Data format
            self.image_height = 256
            self.image_width = 256
            self.channels = 3

            # Train settings
            self.data_divider = 1
            self.n_epochs = 100
            self.n_train = 6000//self.data_divider
            self.n_val = 600//self.data_divider
            self.n_test = 1200//self.data_divider
            self.n_test_in = 600 // self.data_divider
            self.out_frac = (self.n_test-self.n_test_in)/self.n_test
            self.batch_size = 64

            # Data sources
            self.img_folder =   "../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/"
            self.train_folder =   "../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/train/"
            self.val_folder =   "../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/val/"
            self.test_folder = "../weather_detection_data/dreyeve/sunny_highway_countryside_morning_evening_vs_rainy_highway_countryside_morning_evening/test/"
            self.test_in_folder =  self.test_folder + "in/"
            self.test_out_folder =  self.test_folder + "out/"

        if self.dataset == "prosivic":

            self.data_div = 10
            self.n_train = 7000 // self.data_div
            self.n_val = 1400 // self.data_div
            self.n_test = 1000 // self.data_div
            self.n_test_in = 500 // self.data_div
            self.out_frac = (self.n_test-self.n_test_in)/self.n_test
            self.batch_size = 64
            self.num_checkpoints  = 25

            # Data format
            self.image_height = 256
            self.image_width = 256
            self.channels = 3
            self.batch_size = 64

            # Data sources
            self.img_folder =   "../weather_detection_data/prosivic/"
            self.train_folder = "../weather_detection_data/prosivic/train/"
            self.val_folder =   "../weather_detection_data/prosivic/val/"
            self.test_folder = "../weather_detection_data/prosivic/test/"
            self.test_in_folder =  self.test_folder + "in/"
            self.test_out_folder =  self.test_folder + "out/foggy/"


        self.log_dir = './log/'+self.dataset+'/'+self.experiment_name+'/'
        self.model_dir = self.log_dir+'models/'
        self.train_dir = self.log_dir+'train/'
        self.test_dir = self.log_dir+'test/'
