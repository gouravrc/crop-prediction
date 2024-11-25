import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

df = pd.read_csv("Crop_recommendation.csv")
# dropping out 'label' column
data_without_label = df.drop(['label'], axis=1)

# converting 'label' values to numeric
df1 = df.copy()
def convertLabeltoNumeric(df1):
 for i in df1:
  data = df[i].map({
      'rice':1,
      'maize':2,
      'jute':3,
      'cotton':4,
      'papaya':	5,
      'orange':	6,
      'apple':	7,
      'muskmelon':	8,
      'watermelon':	9,
      'grapes':	10,
      'mango'	:11,
      'banana':12,
      'pomegranate':13,
      'lentil':14,
      'blackgram':15,
      'mungbean':16,
      'mothbeans':17,
      'pigeonpeas':18,
      'kidneybeans':19,
      'chickpea':20,
      'coffee':21
      })

 return data

# joining data_without_label dd with numeric label df
fData = data_without_label.join(convertLabeltoNumeric(df))
fData.dropna(inplace=True)

from sklearn.model_selection import train_test_split

X = fData.drop(['label'], axis=1)
y = fData['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
train_data = X_train.join(y_train)

plt.figure(figsize=(10,8))
sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

model = LinearRegression()

X_test
y_test

model.fit(X_train, y_train)

model.score(X_test, y_test)

import sklearn.ensemble as RandomForestRegressor

forest = RandomForestRegressor.RandomForestRegressor()
forest.fit(X_train, y_train)
forest.score(X_test, y_test)
print(forest.score(X_test, y_test))

# new_data = np.array([[66,69,47, 23.69212243, 93.61055571, 6.912299695, 87.53393983]])  # Feed this information from JAVA rest API
# prediction = forest.predict(new_data)
# print(prediction)

with open('cp_model.pkl', 'wb') as f:
  pickle.dump(forest, f)