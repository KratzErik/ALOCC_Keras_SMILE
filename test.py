# Test script for trained ALOCC-model
from utils import *
from kh_tools import *
import models
import imp
imp.reload(models)
from models import ALOCC_Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from configuration import Configuration as cfg
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
#%matplotlib inline

model = ALOCC_Model(dataset_name=cfg.dataset, input_height=cfg.image_height,input_width=image_width, is_training= False)
load_epoch = cfg.load_epoch
self.adversarial_model.load_weights(cfg.model_dir+'checkpoint/ALOCC_Model_%d.h5'%load_epoch)

data = model.data
batch_size = cfg.test_batch_size
n_batches = len(data)//batch_size
model_predicts = []

for batch_idx in range(n_batches):
    batch_data = data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
    batch_predicts = model.adversarial_model.predict(batch_data])
    model_predicts.extend(batch_predicts)

    if cfg.test_batch_verbose:
        batch_labels = model.test_labelsdata[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        # Print metrics for batch
        roc_auc_ = roc_auc_score(batch_labels, batch_predicts)
        print("AUC ")
        pr, rc = precision_recall_curve(batch_labels, batch_predicts)
        prc_auc = auc(rc, pr)

# get final predics
model_predicts.extend(model.adversarial_model.predict(data[n_batches*batch_size:]]))

# Print metricsmodel.test_labels
roc_auc = roc_auc_score(model.test_labels, model_predicts)
print("AUROC:\t", roc_auc)
pr, rc, _ = precision_recall_curve(model.test_labels, model_predicts)
prc_auc = auc(rc, pr)
print("AUPRC:\t", prc_auc)

# Save figures, etc, etc.

