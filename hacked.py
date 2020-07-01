import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score
import numpy as np 


X = pd.read_csv('Train.csv')
X_test = pd.read_csv('Test.csv')
X_sample = pd.read_csv('Test.csv')

X.fillna(X.mean(),inplace=True)
Y = X.MULTIPLE_OFFENSE
X.drop(['INCIDENT_ID','DATE','MULTIPLE_OFFENSE'],axis=1,inplace=True)
X_test.drop(['INCIDENT_ID','DATE'],axis=1,inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

train_X,val_X,train_Y,val_Y = train_test_split(X,Y,train_size=0.8,test_size=0.2,random_state=0)
#model = LinearRegression()
model = RandomForestRegressor(n_estimators=100, random_state=1)
#print(train_X)
model.fit(X,Y)
preds = model.predict(X_test)
output = pd.DataFrame({'INCIDENT_ID':X_sample.INCIDENT_ID,'MULTIPLE_OFFENSE':preds.astype(int)})
output.to_csv('submission.csv', index=False)
# me = mean_absolute_error(val_Y,preds)
# r_score = r2_score(val_Y,preds)
# print(me,r_score)