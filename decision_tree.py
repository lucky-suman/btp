import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier #For Classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.feature_selection import RFE
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

data = pd.read_csv('train.csv')
print 'Initial data size. '
print np.shape(data)

np_data =  np.array(data)

np_new= np.empty([11000, 23])
print 'Preprocessed data size'
print np.shape(np_new)

np_new[:5500, 1: ]= preprocessing.scale(np_data[:5500, 1: ])
np_new[:5500, 0 ]= np_data[:5500, 0]
#print (np_new)
np_new[5500: , 1:12]= preprocessing.scale(np_data[: , 12:23])
#print np_new[5500: , 1:12]
np_new[5500: , 12:23]= preprocessing.scale(np_data[: , 1:12])
#print np_new[5500: , 12:23]
res_ar= np_data[: , 0]
#print np.shape(res_ar)
#print(res_ar)
ar= (np.logical_not(res_ar)).astype(int)
#print ar
np_new[5500: , 0]=ar
#print np_new[5500: , : ]

data_validate= np_new[10800: , 1:]
result_validate = np_new[10800: , 0]

data_train= np_new[:10800, 1:]
data_result= np_new[:10800, 0]
print 'data start to train with Decisiontreeclassifier with RFE and adaboost.'

dt = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
ME')
clf2 = AdaBoostClassifier(n_estimators=50, base_estimator=dt,learning_rate=0.1, algorithm='SAMME')

clf = RFE(clf2, 10)

train= clf.fit(data_train,data_result)
predict_validate=clf.predict(data_validate)
f1score=f1_score(result_validate, predict_validate, average='binary')
print('accuracy is ' ,f1score*100 ) 

#print train.score( data_validate, result_validate )
print 'trained...'
