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
import pickle
from shutil import copyfile
import os
#%matplotlib inline

if __name__ == '__main__':

    # When run as a script, this file loads models trained and saved by running models.py
    # Specify which model to test by providing the same values for arguments --dataset and 
    # --exp_name as when running models.py

    # Both ALOCC novelty scores and autoencoder reconstruction errors are computed for all 
    # test set samples, and corresponding AUROC and AUPRC are printed.
    # There is an option in ./configuration.py to also export the scores and labels to a 
    # specified directory, for post-processing of nd results.

    parser=argparse.ArgumentParser()
    parser.add_argument('--load_epoch','-e', default='final', help='Training epoch to load model from')
    parser.add_argument('--dataset', '-d', default='mnist', help='Dataset to use (overrides configuration)')
    parser.add_argument('--exp_name', '-x', default='debug', help='Name of experiment to load model from')
    parser.add_argument('--out_name', '-o', default=None, help = 'Which folder in ...test/out/ to use as outliers')
    parser.add_argument('--export_results', '-r', default=False, help='If True, scores and labels will be pickled and saved automatically to directory for comparison with other algorithms')
    args=parser.parse_args()
    dataset = args.dataset
    exp_name = args.exp_name
    out_name = args.out_name
    load_epoch = args.load_epoch
    export_results = bool(args.export_results)
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
    log.append("# Dataset: %s, outliers: %s"%(dataset,cfg.test_out_folder.replace(cfg.test_folder,"")))

    model = ALOCC_Model(dataset_name=dataset, input_height=cfg.image_height,input_width=cfg.image_width, is_training= False, outlier_dir = outlier_dir, cfg=cfg)
    if load_epoch == 'final':
        model.load_last_checkpoint()
        load_epoch = str(model.start_epoch-1)
    else:
        trained_adv_path = model_dir+'ALOCC_Model_%s_adv.h5'%load_epoch
        trained_d_path = model_dir+'ALOCC_Model_%s_d.h5'%load_epoch
        model.adversarial_model.load_weights(trained_adv_path)
        model.discriminator.load_weights(trained_d_path)
        
        
    print("Loading trained model from epoch %s"%load_epoch)
    
    data = model.data
    batch_size = model.cfg.test_batch_size
    n_batches = len(data)//batch_size
    scores = np.array([])
    recon_errors = np.array([])
    dscores = np.array([])

    # For compatibility with other algorithm tests, we need label 1 for outliers, 0 for outliers
    # and scores that increase with abnormality => flip labels 0 > 1 and 1 > 0
    model.test_labels = 1 - model.test_labels

    for batch_idx in range(n_batches):
        batch_data = data[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_predicts = model.adversarial_model.predict(batch_data)
        batch_scores = batch_predicts[1]
        batch_recons = batch_predicts[0]

        x = K.eval(binary_crossentropy(K.variable(batch_data), K.variable(batch_recons)))
        batch_recon_errors = x.sum(axis=tuple(range(1,x.ndim)))

        scores = np.append(scores, batch_scores)
        recon_errors = np.append(recon_errors, batch_recon_errors)

        if cfg.use_d_score:
            batch_dscores = model.discriminator(batch_data)
            dscores = np.append(dscores, batch_dscores)

        if model.cfg.test_batch_verbose:
            batch_labels = model.test_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            # Print metrics for batch
            roc_auc_ = roc_auc_score(batch_labels, batch_scores)
            print("AUC ")
            pr, rc = precision_recall_curve(batch_labels, batch_scores)
            prc_auc = auc(rc, pr)
        print("Tested batch %d/%d"%(batch_idx+1,n_batches))
    # All scores computed, evaluate and document

    # get final predics
    if len(data)%batch_size != 0:
        final_data = data[n_batches*batch_size:]
        final_predicts = model.adversarial_model.predict(final_data)
        scores = np.append(scores, final_predicts[1])
        x = K.eval(binary_crossentropy(K.variable(final_data), K.variable(final_predicts[0])))
        final_recon_errors = x.sum(axis=tuple(range(1,x.ndim)))
        recon_errors = np.append(recon_errors, final_recon_errors)

        if cfg.use_d_score:
            dscores = np.append(dscores, model.discriminator[final_data])
            dscores = dscores.max()-dscores

    # For compatibility with other algorithm tests, we need label 1 for outliers, 0 for outliers
    # and scores that increase with abnormality => flip scores so max becomes min, etc.
    scores = scores.max()-scores

    # Save scores and labels for comparison with other experiments
    if export_results:
         # Assert results export dir exists
        if not os.path.exists(cfg.export_results_dir):
            os.makedirs(cfg.export_results_dir)
            print("Created directory %s" % cfg.export_results_dir)

        cfg.test_name = out_name
        def export_scores(score_vector, score_name):
            if cfg.test_name is None:
                results_filepath = cfg.export_results_dir + '%s_ALOCC_%s.pkl'%(dataset, score_name)
                exp_name_file = cfg.export_results_dir + 'experiment_names/%s_ALOCC_%s.txt'%(dataset,score_name)
            else:
                results_filepath = cfg.export_results_dir + '%s_ALOCC_%s_%s.pkl'%(dataset,score_name,cfg.test_name)
                exp_name_file = cfg.export_results_dir + 'experiment_names/%s_ALOCC_%s_%s.txt'%(dataset,score_name,cfg.test_name)

            with open(results_filepath,'wb') as f:
                pickle.dump([score_vector,model.test_labels],f)
            print("Saved D(R(x)) results to %s"%results_filepath)

            # Update score directory log with experiment name
            with open(exp_name_file, 'w') as f:
                f.write(exp_name)

        # First, save actual ALOCC scores: D(R(x))
        export_scores(scores, "DRx")

        # Export recon errors in the same way
        export_scores(recon_errors, "reconerr")

        # Last, save D(x) scores (as compared in ALOCC paper)
        if cfg.use_d_score:
            export_scores(dscores, "Dx")

    # Assert test diagnostic directory exists
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print("Created directory %s" % test_dir)

    # Print metrics
    # D(R(x)) results
    fpr, tpr, _ = roc_curve(model.test_labels, scores, pos_label = 1)
    roc_auc = auc(fpr,tpr)
    print("AUROC D(R(x))-score:\t", roc_auc)
    log.append("# AUROC D(R(x))-score:\t%.5f"%roc_auc)

    pr, rc, _ = precision_recall_curve(model.test_labels, scores, pos_label = 1)
    prc_auc = auc(rc, pr)
    print("AUPRC D(R(x))-score:\t", prc_auc)
    log.append("# AUPRC D(R(x))-score:\t%.5f"%prc_auc)

    if cfg.use_d_score:
        # D(x) results
        fpr, tpr, _ = roc_curve(model.test_labels, dscores, pos_label = 1)
        roc_auc = auc(fpr,tpr)
        print("AUROC D(x)-score:\t", roc_auc)
        log.append("# AUROC D(x)-score:\t%.5f"%roc_auc)

        pr, rc, _ = precision_recall_curve(model.test_labels, dscores, pos_label = 1)
        prc_auc = auc(rc, pr)
        print("AUPRC D(x)-score:\t", prc_auc)
        log.append("# AUPRC D(x)-score:\t%.5f"%prc_auc)

    # Reconstruction error results
    fpr, tpr, _ = roc_curve(model.test_labels, recon_errors, pos_label = 1)
    roc_auc_err = auc(fpr,tpr)
    print("AUROC rec_err:\t", roc_auc_err)
    log.append("# AUROC rec_err:\t%.5f"%roc_auc_err)

    pr, rc, _ = precision_recall_curve(model.test_labels, recon_errors, pos_label = 1)
    prc_auc_err = auc(rc, pr)
    print("AUPRC: rec_err\t", prc_auc_err)
    log.append("# AUPRC: rec_err\t%.5f"%prc_auc_err)

    # Separate inliers and outliers
    inlier_idx = np.where(model.test_labels==0)[0]
    outlier_idx = np.where(model.test_labels==1)[0]

    # Classwise scores
    inlier_scores = scores[inlier_idx]
    outlier_scores = scores[outlier_idx]
    log.insert(2,'#Num inliers: %d'%len(inlier_scores))
    log.insert(2,'#Num outliers: %d'%len(outlier_scores))
    # Sort classes to obtain most normal and anomalous
    in_perm = np.argsort(inlier_scores)
    out_perm = np.argsort(outlier_scores)

    plt.hist(inlier_scores, alpha=0.5, label='Inliers')
    plt.hist(outlier_scores, alpha=0.5, label='Outliers')
    plt.legend(loc='upper right')
    #plt.show()
    print('Saving score histogram to ', test_dir+'scores_hist.png')
    plt.savefig(test_dir+'scores_hist.png')

    # Classwise recon_errors
    inlier_recerr = recon_errors[inlier_idx]
    outlier_recerr = recon_errors[outlier_idx]

    # Recon error histogram
    plt.clf()
    plt.hist(inlier_recerr, alpha=0.5, label='Inliers')
    plt.hist(outlier_recerr, alpha=0.5, label='Outliers')
    plt.legend(loc='upper right')
    print('Saving score histogram to ', test_dir+'recerr_hist.png')
    plt.savefig(test_dir+'recerr_hist.png')

    # Plot reconstructions
    sample_size = 16
    inlier_most_norm_sample = model.data[inlier_idx[in_perm[-sample_size:]]]
    inlier_most_out_sample = model.data[inlier_idx[in_perm[:sample_size]]]
    outlier_most_norm_sample = model.data[outlier_idx[out_perm[-sample_size:]]]
    outlier_most_out_sample = model.data[outlier_idx[out_perm[:sample_size]]]
    all_samples = [inlier_most_norm_sample, inlier_most_out_sample, outlier_most_norm_sample, outlier_most_out_sample]
    sample_names = ['most_normal_inliers', 'most_anomalous_inliers', 'most_normal_outliers', 'most_anomalous_outliers']

    print('Saving reconstruction montages to ', test_dir)
    for sample, name in zip(all_samples, sample_names):
        sample_predicts = model.adversarial_model.predict(sample)
        sample_recon = sample_predicts[0]
        montage_imgs =np.squeeze(np.concatenate([[img1, img2] for img1, img2 in zip(sample, sample_recon)]))
        scipy.misc.imsave(test_dir+name+'_reconstructions.jpg', montage(montage_imgs))

    # add log to configuration file
    with open(log_dir+'configuration.py','a') as outfile: # mode 'a' for append
        for line in log:
            outfile.write(line+'\n')
        
