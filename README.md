# ALOCC
Re-implementation of https://github.com/Tony607/ALOCC_Keras

## How to setup an experiment
The experiment settings are defined in ```./configuration.py```

### Change configuration with implemented dataset
Settings are kept separately for each dataset, under corresponding if-statements for the ```dataset``` variable. The dataset is specified as an argument when running a training or testing session.


### Add a new dataset
* Put your data in separate directories:
    * training data (inliers)
    * validation data (inliers)
    * test data inliers
    * test data outliers
* Do the following in ```configuration.py```:
    * Create a new ```if dataset == "dataset_name"``` block in configuration.py. Copy one of the existing ones so as to not miss any required settings. 
    * Specify the directories in which data is kept in variables ```train_folder```, etc.
    * Make sure number of samples for each set is correct, ```num_train```, etc.
    * Make sure image format corresponds to the image dimensions in your dataset, ```image_height```, ```image_width``` and ```channels```
     * Specify autoencoder and discriminator architectures

## How to train a novelty detection model
* Make sure you have the settings you want in ```configuration.py```
* To train a model with a custom dataset, or the already setup ```prosivic``` and ```dreyeve``` datasets, run the following from the repository main directory:
```
python models.py --dataset dataset_name --exp_name experiment_name
```


The script will train an adversarial model, consisting of an autoencoder and a discriminator on the data in your specified train-directory and store them as .pkl files in ```./log/dataset_name/experiment_name/models/```.

The script takes some optional command line arguments, which are taken from ```configuration.py``` if not specified.

## How to test a novelty detection model
* Make sure you have the settings you want in ```configuration.py```
* Make sure you have trained an adversarial model, see above.
* Run the following from the repository main directory:
```
python test.py --dataset dataset_name --exp_name experiment_name
```
This script also takes some optional command line arguments, which are taken from ```configuration.py``` or set to default values (hardcoded in ```test.py```) if not specified.

This will load the stored discriminator and autoencoder models, compute the ALOCC novelty scores for all test data samples in the specified test data directories, and compute the corresponding AUROC and AUPRC metrics. The same metrics, but using the reconstruction error as novelty score, will also be provided. If the argument```--export_results = 1``` is provided, arrays of scores and test labels (1 for outliers/novelties and 0 for inliers/normal samples) will be saved to the directory specified as ```export_results_dir``` in ```configuration.py```.

## Baseline model
This repository also contains two file called ```baseline_model.py``` and ```baseline_configuration.py```. **They are not needed for the ALOCC model.** They instead define very simple models for comparing images in the test set, such as mean pixel intensity. It was built in order to test how simple the Pro-SiVIC dataset novelty detection task is. (The results are that mean pixel intensity and mean squared intensity achieve almost perfect separation of novelties for that dataset, both urban and foggy scenario.)
