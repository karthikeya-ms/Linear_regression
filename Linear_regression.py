# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 20:49:37 2024

@author: karthikeya_sk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('data.csv')

# plt.xlabel('time')
# plt.ylabel('cells')
# plt.scatter(df.time, df.cells, color='red', marker='+')

x_df = df.drop('cells', axis = 'columns')
y_df = df.cells


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.4, random_state=10)

model = linear_model.LinearRegression()

model.fit(x_train, y_train)

predict_test = model.predict(x_test)
print(y_test, predict_test)




"""
model = linear_model.LinearRegression()
model.fit(x_df, y_df)
print("Predicted no.of cells at time = 2.3 is : ", model.predict([[2.3]]))
print(model.score(x_df, y_df))


c = model.intercept_
m = model.coef_
print('Predicted value is', (m*2.3 + c))

cells_predict_df = pd.read_csv('data_predict.csv') 

predicted_cells = model.predict(cells_predict_df)

cells_predict_df['cells'] = predicted_cells

cells_predict_df.to_csv('data_predictions.csv')

"""