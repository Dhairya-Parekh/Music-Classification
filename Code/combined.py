import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from model1 import MC
from model2 import lstm_model
from feature_extraction import model3_data
from visualize import *

class CombinedModel(nn.Module):
    def __init__(self,model1,model2):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
    
    def forward(self,x):
        x1 = x[0,:].reshape(1,-1)
        x2 = x[1:,:].reshape(-1,10,29)
        return F.softmax(self.model1(x1)) + F.softmax(self.model2(x2,1))
        
def predict(model,x):
    y = model(x)
    return torch.argmax(y)

def test_accuracy(model,x,y):
    y_preds = torch.empty(size=(len(x),))
    with torch.no_grad():
        for i in range(len(x)):
            y_preds[i] = predict(model,x[i])
    metrics(y.tolist(),y_preds.tolist(),'combined')

if __name__ == '__main__':
    model1 = pkl.load(open("../Model/model1.pkl",'rb'))
    model2 = pkl.load(open("../Model/model2.pkl",'rb'))
    labels, trn_x, trn_y, val_x, val_y, tst_x, tst_y = model3_data()
    combined_model = CombinedModel(model1,model2)
    test_accuracy(combined_model,tst_x,tst_y)
    
    
