from PIL import Image,ImageDraw
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
bbox=pd.read_csv("../Data/BBox_List_2017.csv")
data_entry = pd.read_csv("../Data/Data_Entry_2017.csv")

X=[]
y=[]
data_entry['Finding Labels'] = data_entry['Finding Labels'].apply(lambda x: x.split('|')[0])
labels=data_entry["Finding Labels"].unique()

for i in range(100):
    try:
        im=Image.open("../Data/images/"+data_entry["Image Index"][i])
        data = np.array(im,dtype=float)
        if len(data.flatten())==1024*1024:
            X.append(data.flatten())
            y.append(data_entry["Finding Labels"][i])
            print(data.shape)
    except(Exception):
        handle=0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rfc = RandomForestClassifier(
            bootstrap=True,  criterion='entropy',
            max_depth=2, max_features='auto', max_leaf_nodes=None, min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,n_estimators=10, n_jobs=2, oob_score=False,
            random_state=0,verbose=2, warm_start=False
    )
param_dist = {'max_depth': [2, 3, 4],
              'bootstrap': [True],
              'max_features': ['auto', None],
              'criterion': ['gini', 'entropy']}

cv_rf = GridSearchCV(rfc, cv = 2,
                     param_grid=param_dist,
                     n_jobs = 3)
# cv_rf.fit(X_train, y_train)
# print('Best Parameters using grid search: \n',
#       cv_rf.best_params_)
rfc.fit(X_train, y_train)
y_pred=rfc.predict(X_test)


"""
Printing
"""

print(accuracy_score(y_pred,y_test))
for i in range(len(y_pred)):
    print(y_pred[i]," ",y_test[i])
print(accuracy_score(y_pred,y_test))
print(f1_score(y_pred,y_test,average=None))
print(confusion_matrix(y_test,y_pred,labels=labels))
























