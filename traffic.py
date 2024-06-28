import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

dataset = pd.read_csv('traffic.csv')
#print(dataset.head())

dataset["DateTime"]= pd.to_datetime(dataset["DateTime"])
dataset = dataset.drop(["ID"], axis=1) #dropping IDs column
#dataset.info()

#print(dataset.head())

dataframe=dataset.copy()
dataframe["Year"]= dataframe['DateTime'].dt.year
dataframe["Month"]= dataframe['DateTime'].dt.month
dataframe["Date_no"]= dataframe['DateTime'].dt.day
dataframe["Hour"]= dataframe['DateTime'].dt.hour
dataframe["Day"]= dataframe.DateTime.dt.strftime("%A")
#print(dataframe.head())

dataframe = dataframe.drop(["DateTime"], axis =1)
dataframe = dataframe.drop(["Year"], axis=1)

Vehicle = dataframe["Vehicles"]

dataframe = dataframe.drop(["Vehicles"], axis =1)

dataframe["Vehicles"] = Vehicle
#print(dataframe)

X = dataframe.iloc[:, :-1].values  #independent variables
y = dataframe.iloc[:, -1].values   #dependent variables
#print(X)
#print(y)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [-1])], remainder = "passthrough")
X = np.array(ct.fit_transform(X))
#print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
'''print(X_train)
print(X_test)
print(y_train)
print(y_test)'''


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import r2_score
#print(r2_score(y_test, y_pred))

#print(X_test[100])
#print(y_test[100])

'''input_data  = (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4, 4, 6, 16)
input_data_np = np.asarray(input_data)
input_data_reshaped = input_data_np.reshape(1,-1)
prediction = regressor.predict(input_data_reshaped)
prediction = prediction.round()
print(prediction)'''

pickle.dump(regressor,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))