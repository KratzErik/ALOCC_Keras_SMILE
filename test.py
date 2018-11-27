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

model = ALOCC_Model(dataset_name=cfg.dataset, input_height=cfg.image_height,input_width=cfg.image_width, is_training= False)
load_epoch = cfg.load_epoch
model.adversarial_model.load_weights(cfg.model_dir+'checkpoint/ALOCC_Model_%d.h5'%load_epoch)

data = model.data
batch_size = cfg.test_batch_size
n_batches = len(data)//batch_size
scores = np.array([])
recon_errors = np.array([])
for batch_idx in range(n_batches):
    batch_data = data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
    batch_predicts = model.adversarial_model.predict(batch_data)
    batch_scores = batch_predicts[1]
    batch_recons = batch_predicts[0]
    batch_recon_errors = np.array([K.eval(binary_crossentropy(K.variable(img), K.variable(recon))).mean() for img, recon in zip(batch_data, batch_recons)])

    scores = np.append(scores, batch_scores)
    recon_errors = np.append(recon_errors, batch_recon_errors)

    if cfg.test_batch_verbose:
        batch_labels = model.test_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        # Print metrics for batch
        roc_auc_ = roc_auc_score(batch_labels, batch_scores)
        print("AUC ")
        pr, rc = precision_recall_curve(batch_labels, batch_scores)
        prc_auc = auc(rc, pr)

# get final predics
scores = np.append(scores, model.adversarial_model.predict(data[n_batches*batch_size:])[1])

# Assert export dir exists
if not os.path.exists(cfg.test_dir):
    os.makedirs(cfg.test_dir)
    print("Created directory %s" % cfg.test_dir)

# Print metrics
fpr, tpr, _ = roc_curve(model.test_labels, scores, pos_label = 0)
roc_auc = auc(fpr,tpr)
print("AUROC D():\t", roc_auc)
pr, rc, _ = precision_recall_curve(model.test_labels, scores, pos_label = 0)
prc_auc = auc(rc, pr)
print("AUPRC: D()\t", prc_auc)

fpr, tpr, _ = roc_curve(model.test_labels, recon_errors, pos_label = 0)
roc_auc = auc(fpr,tpr)
print("AUROC error:\t", roc_auc)
pr, rc, _ = precision_recall_curve(model.test_labels, recon_errors, pos_label = 0)
prc_auc = auc(rc, pr)
print("AUPRC: error\t", prc_auc)

# Save figures, etc, etc.
inlier_idx = np.where(model.test_labels==1)[0]
outlier_idx = np.where(model.test_labels==0)[0]

# Histogram scores
inlier_scores = scores[inlier_idx]
outlier_scores = scores[outlier_idx]

bins = 100
plt.hist(inlier_scores, bins, alpha=0.5, label='Inliers')
plt.hist(outlier_scores, bins, alpha=0.5, label='Outliers')
plt.legend(loc='upper right')
#plt.show()
plt.savefig(cfg.test_dir+'scores_hist.png')

# Plot some inliers with reconstructions
sample_size = 32
inlier_sample = model.data[np.random.choice(inlier_idx, sample_size//2)]
outlier_sample = model.data[np.random.choice(outlier_idx, sample_size//2)]
sample = np.concatenate([inlier_sample, outlier_sample])

# Plot some outliers with reconstructions
sample_predicts = model.adversarial_model.predict(sample)
sample_recon = sample_predicts[0]
sample_scores = sample_predicts[1]
montage_imgs =np.squeeze(np.concatenate([[img1, img2] for img1, img2 in zip(sample, sample_recon)]))
scipy.misc.imsave(cfg.test_dir+'test_reconstruction_samples.jpg', montage(montage_imgs))
print("Sample scores:")
print(sample_scores)

