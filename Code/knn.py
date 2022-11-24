from sklearn.neighbors import KNeighborsClassifier
from feature_extraction import model1_data, model2_data
from visualize import *

labels, trn_x, trn_y, val_x, val_y, tst_x, tst_y = model1_data() 

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(trn_x, trn_y)
knn.fit(trn_x, trn_y)
Y_pred=knn.predict(val_x).tolist()               
metrics(val_y,Y_pred,'KNN')

# print(f"knn_trn : {((knn.predict(trn_x)-trn_y.cpu().detach().numpy())==0).sum()/(trn_y.shape[0])}")
# print(f"knn_val : {((knn.predict(val_x)-val_y.cpu().detach().numpy())==0).sum()/(val_y.shape[0])}")
# print(f"knn_tst : {((knn.predict(tst_x)-tst_y.cpu().detach().numpy())==0).sum()/(tst_y.shape[0])}")

# print(neigh.predict_proba([[0.9]]))