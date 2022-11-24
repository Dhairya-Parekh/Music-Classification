import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from feature_extraction import model1_data, model2_data
from visualize import *

labels, trn_x, trn_y, val_x, val_y, tst_x, tst_y = model1_data() 

svm_models = {
    'linear' : make_pipeline(StandardScaler(), SVC(kernel='linear', C=1)),
    'poly' : make_pipeline(StandardScaler(), SVC(kernel='poly',degree=3, coef0=1, C=1)),
    'rbf' : make_pipeline(StandardScaler(), SVC(kernel='rbf',gamma='auto', C=5)),
    'sigmoid' : make_pipeline(StandardScaler(), SVC(kernel='sigmoid', gamma='auto', C=1)),
}
print("*"*10)
for kernel, model in svm_models.items():
    print(kernel)
    model.fit(trn_x, trn_y)
    Y_pred=model.predict(val_x).tolist()               
    metrics(val_y,Y_pred,'SVM')

    # print(f"{kernel}_trn : {((model.predict(trn_x)-trn_y.cpu().detach().numpy())==0).sum()/(trn_y.shape[0])}")
    # print(f"{kernel}_val : {((model.predict(val_x)-val_y.cpu().detach().numpy())==0).sum()/(val_y.shape[0])}")
    # print(f"{kernel}_tst : {((model.predict(tst_x)-tst_y.cpu().detach().numpy())==0).sum()/(tst_y.shape[0])}")
    print("*"*10)
