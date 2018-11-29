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
from configuration import Configuration
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import argparse
import datetime
#%matplotlib inline

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--load_epoch','-e', default='final', help='Training epoch to load model from')
    parser.add_argument('--dataset', '-d', default='mnist', help='Dataset to use (overrides configuration)')
    parser.add_argument('--exp_name', '-x', default='debug', help='Name of experiment to load model from')
    parser.add_argument('--out_name', '-o', default=None, help = 'Which folder in ...test/out/ to use as outliers')

    args=parser.parse_args()
    dataset = args.dataset
    exp_name = args.exp_name
    out_name = args.out_name
    load_epoch = args.load_epoch
    log_dir = 'log/'+dataset+'/'+exp_name+'/'
    model_dir = log_dir + 'models/'
    test_dir =  log_dir + 'test/'

    cfg = Configuration(dataset, exp_name)

    if args.out_name is None:
        outlier_dir = cfg.test_out_folder
    else:
        outlier_dir = cfg.test_folder + "out/" + args.out_name + "/"

    log = ["################################################################"]
    test_time = datetime.datetime.now()
    log.append("# Test started at: %s"%test_time)

    trained_model_path = model_dir+'ALOCC_Model_%s.h5'%load_epoch
    print("Loading trained model from %s"%trained_model_path)
    model = ALOCC_Model(dataset_name=dataset, input_height=cfg.image_height,input_width=cfg.image_width, is_training= False, outlier_dir = outlier_dir, cfg=cfg)
    model.adversarial_model.load_weights(trained_model_path)

    data = model.data
    batch_size = model.cfg.test_batch_size
    n_batches = len(data)//batch_size
    scores = np.array([])
    # recon_errors = np.array([])

    # NOTE: below, scores and true labels are inverted so that 0 <- 1, and 1 <- 0, to have 1 for the positive class, which is outliers
#    scores = 1-scores
#    model.test_labels = 1-model.test_labels

    for batch_idx in range(n_batches):
        batch_data = data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_predicts = model.adversarial_model.predict(batch_data)
        batch_scores = batch_predicts[1]
        batch_recons = batch_predicts[0]
    #    batch_recon_errors = np.array([K.eval(binary_crossentropy(K.variable(img), K.variable(recon))).mean() for img, recon in zip(batch_data, batch_recons)])

        scores = np.append(scores, batch_scores)
    #    recon_errors = np.append(recon_errors, batch_recon_errors)

        if model.cfg.test_batch_verbose:
            batch_labels = model.test_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            # Print metrics for batch
            roc_auc_ = roc_auc_score(batch_labels, batch_scores)
            print("AUC ")
            pr, rc = precision_recall_curve(batch_labels, batch_scores)
            prc_auc = auc(rc, pr)

    # All scores computed, evaluate and document

    # get final predics
    scores = np.append(scores, model.adversarial_model.predict(data[n_batches*batch_size:])[1])

    # Assert export dir exists
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print("Created directory %s" % test_dir)

    # Print metrics
    fpr, tpr, _ = roc_curve(model.test_labels, -scores, pos_label = 0)
    roc_auc = auc(fpr,tpr)
    print("AUROC D()-score:\t", roc_auc)
    log.append("AUROC D()-score:\t%.5f"%roc_auc)
    
    pr, rc, _ = precision_recall_curve(model.test_labels, -scores, pos_label = 0)
    prc_auc = auc(rc, pr)
    print("AUPRC D()-score:\t", prc_auc)
    log.append("AUPRC D()-score:\t%.5f"%prc_auc)

    #fpr, tpr, _ = roc_curve(model.test_labels, recon_errors, pos_label = 0)
    #roc_auc = auc(fpr,tpr)
    #print("AUROC error:\t", roc_auc)
    #pr, rc, _ = precision_recall_curve(model.test_labels, recon_errors, pos_label = 0)
    #prc_auc = auc(rc, pr)
    #print("AUPRC: error\t", prc_auc)

    # Save figures, etc, etc.
    inlier_idx = np.where(model.test_labels==1)[0]
    outlier_idx = np.where(model.test_labels==0)[0]

    # Classwise scores
    inlier_scores = scores[inlier_idx]
    outlier_scores = scores[outlier_idx]

    # Sort classes to obtain most normal and anomalous
    in_perm = np.argsort(inlier_scores)
    out_perm = np.argsort(outlier_scores)

    bins = 100
    plt.hist(inlier_scores, bins, alpha=0.5, label='Inliers')
    plt.hist(outlier_scores, bins, alpha=0.5, label='Outliers')
    plt.legend(loc='upper right')
    #plt.show()
    print('Saving score histogram to ', test_dir+'scores_hist.png')
    plt.savefig(test_dir+'scores_hist.png')

    # Plot reconstructions
    sample_size = 16
    inlier_most_norm_sample = model.data[inlier_idx[in_perm[:sample_size]]]
    inlier_most_out_sample = model.data[inlier_idx[in_perm[-sample_size:]]]
    outlier_most_norm_sample = model.data[outlier_idx[out_perm[:sample_size]]]
    outlier_most_out_sample = model.data[outlier_idx[out_perm[-sample_size:]]]
    all_samples = [inlier_most_norm_sample, inlier_most_out_sample, outlier_most_norm_sample, outlier_most_out_sample]
    sample_names = ['most_normal_inliers', 'most_anomalous_inliers', 'most_normal_outliers', 'most_anomalous_outliers']

    print('Saving reconstruction montages to ', test_dir)
    for sample, name in zip(all_samples, sample_names):
        sample_predicts = model.adversarial_model.predict(sample)
        sample_recon = sample_predicts[0]
        montage_imgs =np.squeeze(np.concatenate([[img1, img2] for img1, img2 in zip(sample, sample_recon)]))
        scipy.misc.imsave(test_dir+name+'_reconstructions.jpg', montage(montage_imgs[:,:,:,:]))

    # add log to configuration file

    with open(log_dir+'configuration.py','w') as outfile:
        for line in log:
            outfile.write(line + '\n')
        
