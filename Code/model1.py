'''
Feed forward Neural Network for Music Genre Classification AI-ML Project
Dhairya Parekh(200050097)| Utkarsh Pratap Singh(200050146)| Naman Singh Rana(200050083)| Aditya Kadoo(200050055)
'''
import pandas as pd
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import pickle as pkl
from visualize import *
from feature_extraction import model1_data

class MC(nn.Module):
    def __init__(self,input_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size,55)
        self.fc2 = nn.Linear(55,34)
        self.fc3 = nn.Linear(34,10)
        self.fc4 = nn.Linear(20,10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.drop1 = nn.Dropout(0.05)
        self.drop2 = nn.Dropout(0.05)
        self.m = nn.LeakyReLU(0.1)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

        
    def forward(self,x):
        # x = self.fc1(x)
        # x = self.tanh(x)
        # # x = self.drop1(x)
        # x = self.fc2(x)
        # x = self.tanh(x)
        # # x = self.drop2(x)
        # x = self.fc3(x)
        # # x = self.tanh(x)
        x = self.fc1(x)
        x= self.tanh(x)
        # x = self.drop1(x)
        x = self.fc2(x)
        x = self.tanh(x)
        # x = self.drop2(x)
        x = self.fc3(x)
        
        # x = self.tanh(x)

        return x


'''      
Training neural network
'''
def train(model,X_trn,Y_trn,X_val,Y_val,epochs=80):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = torch.nn.CrossEntropyLoss()
    count = len(X_trn)
    test_err=[]
    for epoch in range(epochs):
        running_loss = 0
        for i  in range(count):
            optimizer.zero_grad()
            v = model(X_trn[i])
            loss = loss_func(v,Y_trn[i])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        test_err.append(running_loss)
        print("epoch: ",epoch," loss: ",running_loss)
    plot_loss(test_err,'model1')   
    y_preds = torch.empty(size=(len(X_val),))
    with torch.no_grad():
        for i in range(len(X_val)):
            v = model(X_val[i])
            y_preds[i] = torch.argmax(v)
    metrics(Y_val.tolist(),y_preds.tolist(),'model1')
    
       
    
    
if __name__ == '__main__':
    
    labels, trn_x, trn_y, val_x, val_y, tst_x, tst_y = model1_data()    
    model = MC(trn_x.shape[1])
    train(model,trn_x,trn_y,val_x,val_y,100)
    
    # save the model
    pkl.dump(model,open("../Model/model1.pkl","wb"))    
    