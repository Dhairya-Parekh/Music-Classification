import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import pickle as pkl
from feature_extraction import model2_data
from visualize import *

torch.manual_seed(0)
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
        
        self.lstm = nn.LSTM(input_size=input_dim,hidden_size=hidden_size, num_layers = num_layers,batch_first=True)
        self.fc1 = nn.Linear(hidden_size,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50,50)
        self.fc4 = nn.Linear(50,output_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)
                                
        
    def forward(self,x,prev_state):
        out,state = self.lstm(x,prev_state)
        out = self.fc1(out[:,-1,:])
        out = self.relu(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = self.fc4(out)
        return out,state

    def init(self,batch_size):
        return (torch.randn(self.num_layers, batch_size, self.hidden_size),
                torch.randn(self.num_layers, batch_size, self.hidden_size))
    
def train(model,train_dataloader,val_dataloader,epochs=80,batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = torch.nn.CrossEntropyLoss()
    test_err=[]
    for epoch in range(epochs):
        running_loss = 0
        hx,cx = model.init(batch_size)
        for x,y in train_dataloader:
            optimizer.zero_grad()
            v,(hx,cx) = model(x,(hx,cx))
            loss = loss_func(v,y)
            hx = hx.detach()
            cx = cx.detach()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        test_err.append(running_loss)
        print("epoch: ",epoch," loss: ",running_loss)
    plot_loss(test_err,'LSTM')   
    Y_val=[]
    Y_pred=[]
    with torch.no_grad():
        hx,cx = model.init(batch_size)
        for x,y in val_dataloader:
            v,(hx,cx) = model(x,(hx,cx))
            Y_val+=y.tolist()
            y_preds = torch.argmax(v,axis=1)
            Y_pred+=y_preds.tolist()
            
            
    metrics(Y_val,Y_pred,'LSTM')
    
            
        
if __name__ == '__main__':
    labels, trn_x, trn_y, val_x, val_y, tst_x, tst_y = model2_data()
    train_dataset = MusicDataset(trn_x,trn_y)
    val_dataset = MusicDataset(val_x,val_y)
    batch_size = 64
    train_dataloader = DataLoader(train_dataset,batch_size,drop_last=True)
    val_dataloader = DataLoader(val_dataset,batch_size,drop_last=True)
    
    model = lstm_model(input_dim=trn_x.shape[2],hidden_size=100,output_size=10,num_layers=3)
    train(model,train_dataloader,val_dataloader,200,batch_size)
    pkl.dump(model,open("../Model/lstm.pkl","wb"))
