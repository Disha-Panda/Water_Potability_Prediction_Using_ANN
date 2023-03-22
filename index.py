# ************WATER POTABILITY PREDICTION USING ARTIFICIAL NEURAL NETWORK ***********


# IMPORTING REQUIRED LIABRARIES

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt


# RENDORING CSV DATA

data = pd.read_csv("Water_potability.csv")
print(data.head())


# sns.heatmap(data)
# plt.show()


# DELETING THE ROWS HAVING NULL VALUES

clean_data = data.dropna()
# sns.heatmap(
# plt.show()


# sns.pairplot(clean_data,hue = "Potability")
# plt.show()


# SPLITTING FEATUTRES AND TARGETS

x = clean_data.drop(columns=['Potability'])
y = clean_data.Potability

# print(x)

# print(y)


# SPLITTING TRAINING AND TEST DATA

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)


# SCALING THE FEATURED VALUES

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# IMPORTING REQUIRED LIABRAIES FOR CREATING ANN ARCHITECTURE

import tensorflow 
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense


# DEVELOPING A NEURAL NETWORK ARCHITECTURE

model = Sequential()

model.add(Dense(64,activation='relu',input_dim=9))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation= 'relu'))
model.add(Dense(1,activation = 'sigmoid'))


# model.summary()


# SETTING THE MODELS LOSS FUNCTION AND OPTIMIZER

model.compile(loss="binary_crossentropy",optimizer="Adam",metrics =["accuracy"])


# TRAINING OUR MODEL USING TRAINING DATA SET

history = model.fit(x_train_scaled,y_train,epochs=15,validation_split=0.2)

logy = model.predict(x_test_scaled)
y_pred = np.where(logy>0.5,1,0)


# CALCULATING THE ACCURACY

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))


# PLOTTING LOSS AND VALIDATION LOSS

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()