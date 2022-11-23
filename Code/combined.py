import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from model1 import MC
from model2 import lstm_model

class CombinedModel(nn.Module):
    def __init__(self,model1,model2):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
    
    def forward(self,x):
        return F.softmax(self.model1(x)) + F.softmax(self.model2(x))
    
def predict(model,x):
    y = model(x)
    return torch.argmax(y)

def test_accuracy(model,x,y):
    y_preds = torch.empty(size=(len(x),))
    with torch.no_grad():
        for i in range(len(x)):
            y_preds[i] = predict(model,x)
    print("accuracy: ",sum(y_preds==y)/len(y))

if __name__ == '__main__':
    model1 = pkl.load(open("../Model/model1.pkl",'rb'))
    model2 = pkl.load(open("../Model/model2.pkl",'rb'))
    
    combined_model = CombinedModel(model1,model2)
    test_accuracy(model,X_test,Y_test)
    

