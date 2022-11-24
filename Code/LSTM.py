import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import pickle as pkl
from feature_extraction import model2_data
from visualize import *

class MusicDataset(Dataset):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

class lstm_model(nn.Module):
    def __init__(self,input_dim,hidden_size,output_size,num_layers) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        self.lstm = nn.LSTM(input_size=input_dim,hidden_size=hidden_size, num_layers = num_layers,dropout=0.01)
        self.fc1 = nn.Sequential(nn.Linear(hidden_size,50),
                                 nn.Tanh(),)
        self.fc2 = nn.Sequential(nn.Linear(50,50),
                                 nn.Tanh(),)
        self.fc3 = nn.Sequential(nn.Linear(50,25),
                                )
        self.fc4 = nn.Sequential(nn.Linear(25,10),
                                 )
        
        
    
    def forward(self,x):
        #  Turn (batch_size X seq_len X input_dim) back into (seq_len x batch_size x hidden_size)
        x = x.transpose(0,1)
        h_0 = Variable(torch.zeros(self.num_layers,batch_size,self.hidden_size)) 
        c_0 = Variable(torch.zeros(self.num_layers,batch_size,self.hidden_size))
        out,_ = self.lstm(x,(h_0,c_0))
        out = self.fc1(out[-1,:,:])
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out


def train(model,train_dataloader,val_dataloader,epochs=80,batch_size=16):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_func = torch.nn.CrossEntropyLoss()
    test_err=[]
    for epoch in range(epochs):
        running_loss = 0
        for x,y in train_dataloader:
            optimizer.zero_grad()
            v = model(x)
            loss = loss_func(v,y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            test_err.append(running_loss)
        print("epoch: ",epoch," loss: ",running_loss)
    plot_loss(test_err,'LSTM')   
    Y_val=[]
    Y_pred=[]
    with torch.no_grad():
        for x,y in val_dataloader:
            v = model(x)
            Y_val+=y.tolist()
            y_preds = torch.argmax(v,axis=1)
            Y_pred+=y_preds.tolist()
            
            
    metrics(Y_val,Y_pred,'LSTM')
    
            
        
if __name__ == '__main__':
    labels, trn_x, trn_y, val_x, val_y, tst_x, tst_y = model2_data()
    train_dataset = MusicDataset(trn_x,trn_y)
    val_dataset = MusicDataset(val_x,val_y)
    batch_size = 16
    train_dataloader = DataLoader(train_dataset,batch_size,drop_last=True)
    val_dataloader = DataLoader(val_dataset,batch_size,drop_last=True)
    
    model = lstm_model(input_dim=trn_x.shape[2],hidden_size=100,output_size=10,num_layers=3)
    train(model,train_dataloader,val_dataloader,80,batch_size)
    pkl.dump(model,open("../Model/lstm.pkl","wb"))
