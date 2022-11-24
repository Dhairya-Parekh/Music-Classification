from sklearn.neighbors import KNeighborsClassifier
from feature_extraction import model1_data, model2_data
from visualize import *

labels, trn_x, trn_y, val_x, val_y, tst_x, tst_y = model1_data() 

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(trn_x, trn_y)
knn.fit(trn_x, trn_y)
Y_pred=knn.predict(val_x).tolist()               
metrics(val_y,Y_pred,'KNN')
