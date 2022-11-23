import pandas as pd
import torch
import numpy as np
import random

TRN = 0.8
VAL = 0.1
TST = 0.1
RAND_SEED = 0

def model1_data():
    df = pd.read_csv("../Dataset/features_30_sec.csv")
    
    df.drop(['filename','length'],axis=1,inplace=True)
    cols = df.columns
    
    for col in cols:
        if col.endswith("var"):
            df.drop([col],axis=1,inplace=True)
    
    df = df.sample(frac=1, random_state=RAND_SEED)

    # normalization
    Y = df['label']
    df.drop(['label'],axis=1,inplace=True)
    mean = df.mean(axis=0)
    sd = df.std(axis=0)
    df = (df-mean)/sd

    labels = Y.unique()
    label_idx = {label:i for i,label in enumerate(labels)}
    label_count = {label:(Y == label).sum() for label in labels}
    used_count = {label:0 for label in labels}

    trn_x = []
    trn_y = []
    val_x = []
    val_y = []
    tst_x = []
    tst_y = []
    
    for index, row in df.iterrows():
        row_label = Y.loc[index]
        used_count[row_label] += 1
        if used_count[row_label] <= TRN*label_count[row_label]:
            trn_y.append(label_idx[row_label])
            trn_x.append([row[col] for col in cols if col!='label' and not col.endswith("var")])
        elif used_count[row_label] <= (TRN+VAL)*label_count[row_label]:
            val_y.append(label_idx[row_label])
            val_x.append([row[col] for col in cols if col!='label' and not col.endswith("var")])
        elif used_count[row_label] <= (TRN+VAL+TST)*label_count[row_label]:
            tst_y.append(label_idx[row_label])
            tst_x.append([row[col] for col in cols if col!='label' and not col.endswith("var")])
    
    trn_x = torch.tensor(trn_x,dtype=torch.float32)
    trn_y = torch.tensor(trn_y)
    val_x = torch.tensor(val_x,dtype=torch.float32)
    val_y = torch.tensor(val_y)
    tst_x = torch.tensor(tst_x,dtype=torch.float32)
    tst_y = torch.tensor(tst_y)

    return labels, trn_x, trn_y, val_x, val_y, tst_x, tst_y

def model2_data():
    df = pd.read_csv("../Dataset/features_3_sec.csv")
    
    df.drop(['filename','length'],axis=1,inplace=True)
    cols = df.columns
    
    for col in cols:
        if col.endswith("var"):
            df.drop([col],axis=1,inplace=True)
    
    df = df.sample(frac=1, random_state=RAND_SEED)

    # normalization
    Y = df['label'][df.index%10==0]
    df.drop(['label'],axis=1,inplace=True)
    X = df.to_numpy().reshape(len(Y),10,-1)
    mean = df.mean(axis=0)
    sd = df.std(axis=0)
    df = (df-mean)/sd

    labels = Y.unique()
    label_idx = {label:i for i,label in enumerate(labels)}
    label_count = {label:(Y == label).sum() for label in labels}
    used_count = {label:0 for label in labels}

    trn_x = []
    trn_y = []
    val_x = []
    val_y = []
    tst_x = []
    tst_y = []
    
    for index, row in enumerate(X):
        row_label = Y.loc[index*10]
        used_count[row_label] += 1
        if used_count[row_label] <= TRN*label_count[row_label]:
            trn_y.append(label_idx[row_label])
            trn_x.append(row)
        elif used_count[row_label] <= (TRN+VAL)*label_count[row_label]:
            val_y.append(label_idx[row_label])
            val_x.append(row)
        elif used_count[row_label] <= (TRN+VAL+TST)*label_count[row_label]:
            tst_y.append(label_idx[row_label])
            tst_x.append(row)
    
    trn_x = torch.tensor(np.array(trn_x),dtype=torch.float32)
    trn_y = torch.tensor(np.array(trn_y))
    val_x = torch.tensor(np.array(val_x),dtype=torch.float32)
    val_y = torch.tensor(np.array(val_y))
    tst_x = torch.tensor(np.array(tst_x),dtype=torch.float32)
    tst_y = torch.tensor(np.array(tst_y))
    
    
    return labels, trn_x, trn_y, val_x, val_y, tst_x, tst_y

def model3_data():
    df1 = pd.read_csv("../Dataset/features_30_sec.csv")
    
    df1.drop(['filename','length'],axis=1,inplace=True)
    cols1 = df1.columns
    
    for col in cols1:
        if col.endswith("var"):
            df1.drop([col],axis=1,inplace=True)
    
    df2 = pd.read_csv("../Dataset/features_3_sec.csv")
    
    df2.drop(['filename','length'],axis=1,inplace=True)
    cols2 = df2.columns
    
    for col in cols2:
        if col.endswith("var"):
            df2.drop([col],axis=1,inplace=True)
    
    #normalization 1
    Y1 = df1['label']
    df1.drop(['label'],axis=1,inplace=True)
    mean = df1.mean(axis=0)
    sd = df1.std(axis=0)
    df1 = (df1-mean)/sd

    # normalization 2
    Y2 = df2['label'][df2.index%10==0]
    df2.drop(['label'],axis=1,inplace=True)
    X2 = df2.to_numpy().reshape(len(Y2),10,-1)
    mean = df2.mean(axis=0)
    sd = df2.std(axis=0)
    df2 = (df2-mean)/sd

    labels = Y1.unique()
    label_idx = {label:i for i,label in enumerate(labels)}
    label_count = {label:(Y1 == label).sum() for label in labels}
    used_count = {label:0 for label in labels}

    trn_x = []
    trn_y = []
    val_x = []
    val_y = []
    tst_x = []
    tst_y = []

    random.seed(RAND_SEED)
    idx = list(range(len(X2)))
    random.shuffle(idx)
    
    for index in idx:
        row_label = Y1.loc[index]
        row1 = np.array(df1.loc[index])
        row2 = X2[index]
        used_count[row_label] += 1
        if used_count[row_label] <= TRN*label_count[row_label]:
            trn_y.append(label_idx[row_label])
            trn_x.append(np.vstack((row1[None, :],row2)))
        elif used_count[row_label] <= (TRN+VAL)*label_count[row_label]:
            val_y.append(label_idx[row_label])
            val_x.append(np.vstack((row1[None, :],row2)))
        elif used_count[row_label] <= (TRN+VAL+TST)*label_count[row_label]:
            tst_y.append(label_idx[row_label])
            tst_x.append(np.vstack((row1[None, :],row2)))
    
    trn_x = torch.tensor(np.array(trn_x),dtype=torch.float32)
    trn_y = torch.tensor(np.array(trn_y))
    val_x = torch.tensor(np.array(val_x),dtype=torch.float32)
    val_y = torch.tensor(np.array(val_y))
    tst_x = torch.tensor(np.array(tst_x),dtype=torch.float32)
    tst_y = torch.tensor(np.array(tst_y))
    
    return labels, trn_x, trn_y, val_x, val_y, tst_x, tst_y
