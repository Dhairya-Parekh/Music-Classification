import pandas as pd
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import pickle as pkl
from feature_extraction import model1_data
from torch.utils.data import Dataset,DataLoader
from visualize import *

class MusicDataset(Dataset):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class FFN(nn.Module):
    def __init__(self,input_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size,200)
        self.fc2 = nn.Linear(200,100)
        self.fc3 = nn.Linear(100,50)
        self.fc4 = nn.Linear(50,10)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop1 = nn.Dropout(0.05)
        self.drop2 = nn.Dropout(0.05)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)

        
    def forward(self,x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.tanh(x)
        x = self.fc4(x)
        return x


def train(model,trn_dataloader,val_dataloader,epochs=80):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = torch.nn.CrossEntropyLoss()
    test_err=[]
    for epoch in range(epochs):
        running_loss = 0
        for x,y  in trn_dataloader:
            optimizer.zero_grad()
            v = model(x)
            loss = loss_func(v,y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        test_err.append(running_loss)
        print("epoch: ",epoch," loss: ",running_loss)
    plot_loss(test_err,'FFN')
    Y_pred=[]
    Y_val=[]   
    with torch.no_grad():
        for x,y in val_dataloader:
            v = model(x)
            Y_val+=y.tolist()
            y_preds = torch.argmax(v,axis=1)
            Y_pred+=y_preds.tolist()
                    
    metrics(Y_val,Y_pred,'FFN')
    
              
def predict(model,x):
    y = model(x)
    return torch.argmax(y)

def test_accuracy(model,x,y):
    y_preds = torch.empty(size=(len(x),))
    with torch.no_grad():
        for i in range(len(x)):
            y_preds[i] = predict(model,x[i])
    print("accuracy: ",sum(y_preds==y)/len(y))
    
if __name__ == '__main__':
    
    batch_size = 4
    labels, trn_x, trn_y, val_x, val_y, tst_x, tst_y = model1_data()    
    trn_dataset = MusicDataset(trn_x,trn_y)
    val_dataset = MusicDataset(val_x,val_y)
    trn_dataloader = DataLoader(trn_dataset,batch_size)
    val_dataloader = DataLoader(val_dataset,batch_size)
    model = FFN(trn_x.shape[1])
    train(model,trn_dataloader,val_dataloader,50)
    test_accuracy(model,tst_x,tst_y)
    test_accuracy(model,trn_x,trn_y)
    pkl.dump(model,open("../Model/ffn.pkl","wb"))    
    