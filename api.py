import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

data=pd.read_csv('housing.csv')

data = data.fillna(data.median())

data['rooms_per_bedrooms']=data['total_rooms']/data['total_bedrooms']

data['rooms_per_bedrooms']=data.apply(lambda row: row.total_rooms / row.total_bedrooms, axis=1)

data.corr()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data['ocean_proximity'] = le.fit_transform(data.ocean_proximity.values).astype(np.float64)

x=data.iloc[:,[0,1,2,3,4,5,6,7,9,10]].values
y=data.iloc[:,8].values

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[8])],remainder='passthrough')
# x=np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn import metrics
from sklearn.model_selection import cross_val_score

def cross_val(model):
    pred = cross_val_score(model, x, y, cv=8)
    return pred.mean()

def print_evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)

def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

"""# 1.8 Random Forest Regressor"""

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=20)
rf_reg.fit(x_train, y_train)

test_pred = rf_reg.predict(x_test)
train_pred = rf_reg.predict(x_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

print('Train set evaluation:\n_____________________________________')
print_evaluate(y_train, train_pred)

# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(rf_reg, open('model.pkl','wb'))

'''
#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2.6, 8, 10.1]]))
'''
