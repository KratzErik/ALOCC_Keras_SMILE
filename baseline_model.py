from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from configuration import Configuration as cfg'
from sklearn.filters.rank import entropy
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

class baseline_model():
    def __init__(self, ad_score_type = 'mean', dataset = 'prosivic'):

        self.dataset = dataset
        self.dataset_choices = ['dreyeve', 'prosivic']
        self.ad_score_type = ad_score_type # choices
        self.score_choices = ['mean','mean_squared']
        self.scores = []
        self.data = []
        self.labels = []

        self.load_test_data() # no training required for this method
        self.score_images()
        self.check_assertions()

    def check_assertions(self):
        assert(self.dataset in self.dataset_choices)
        assert(len(self.data) > 0)
        assert(len(self.scores) == len(self.labels))
        assert(self.ad_score_type in self.score_choices)

    def load_test_data(self):
        n_test_out = cfg.n_test - cfg.n_test_in
        X_test_in = np.array([img_to_array(load_img(cfg.test_in_folder + filename)) for filename in os.listdir(cfg.test_in_folder)][:cfg.n_test_in])
        X_test_out = np.array([img_to_array(load_img(cfg.test_out_folder + filename)) for filename in os.listdir(cfg.test_out_folder)][:n_test_out])
        y_test_in  = np.zeros((len(X_test_in),),dtype=np.int32)
        y_test_out = np.ones((len(X_test_out),),dtype=np.int32)
        self.data = np.concatenate([X_test_in, X_test_out]) / 255.0
        print("Loaded data: %d inliers, %d outliers"%(len(X_test_in),len(X_test_out)))
        self.labels = np.concatenate([y_test_in, y_test_out])
        print("Loaded labels: %d values"%len(labels))

    def score_images(self):
        if self.ad_score_type == 'mean':
            self.scores = [img.mean() for img in self.data]

        elif self.ad_score_type == 'mean_squared':
            self.scores = [np.square(img).mean() for img in self.data]

    def test(self)
        print("Testing dataset %s with score function %s"%(self.dataset, self.ad_score_type))
        
        # Compute roc_auc
        fpr, tpr, _ = roc_curve(self.labels, self.scores)
        roc_auc = auc(fpr,tpr)
        print("AUROC:\t%.5f"%roc_auc)

        # Compute roc_auc
        pr, rc = roc_curve(self.labels, self.scores)
        roc_prc = auc(rc,pr)
        print("AUPRC:\t%.5f"%roc_prc)




if __name__ == '__main__': # executed when file is ran with python3 energy_density_baseline.py
    for score_type in ['mean', 'mean_squared']:
        model = baseline_model(ad_score_type = score_type, dataset = 'prosivic')
        model.test()
    


