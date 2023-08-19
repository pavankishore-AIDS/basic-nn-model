# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons. In this experiment, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

Build your training and test set from the dataset, here we are making the neural network 3 hidden layer with activation layer.as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![](network%20model.PNG)

## DESIGN STEPS

- STEP 1 : Load the dataset fron authorizing the access to drive or directly ope using read.csv
- STEP 2 : Split the dataset into training and testing
- STEP 3 : Create MinMaxScalar objects ,fit the model and transform the data.
- STEP 4 : Build the Neural Network Model and compile the model.
-  STEP 5 : Train the model with the training data.
- STEP 6: Plot the performance plot
- STEP 7: Evaluate the model with the testing data.

## PROGRAM
```
Developed by: Kaushika A
Register no: 212221230048
```
```python
# reading the data file
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds,_=default()
gc=gspread.authorize(creds)

sheet=gc.open('exp 1 dataset').sheet1
data=sheet.get_all_values()
```
```python
# importing packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
```
```python
# creating dataframe
df=pd.DataFrame(data[1:],columns=data[0])
df=df.astype({'input':'float'})
df=df.astype({'output':'float'})
df.head()
```
```python
# creating train & test data
X=df[['input']].values
y=df[['output']].values

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=33)
```
```python
# scaling the data
scaler=MinMaxScaler()
scaler.fit(x_train)

x_trained=scaler.transform(x_train)
```
```python
#creating and compiling the network model
ai_brain = Sequential([
    Dense(units = 10, activation = 'relu', input_shape=[1]),
    Dense(units = 19,activation = 'relu'),
    Dense(1)
])

ai_brain.compile(optimizer = 'rmsprop' , loss = 'mse')
```
```python
# training the network model
ai_brain.fit(x_trained,y_train,epochs=2000)
```
```python
# plotting loss graph
loss_df = pd.DataFrame(ai_brain.history.history)

loss_df.plot()
```
```python
# find RMSE of the network model
X_test1 = scaler.transform(x_test)

ai_brain.evaluate(X_test1,y_test)
```
```python
# predicting using the network model
X_n1=[[4]]

X_n1_1=scaler.transform(X_n1)

ai_brain.predict(X_n1_1)
```


## Dataset Information

![](dataset.PNG)

## OUTPUT

### Training Loss Vs Iteration Plot

![](plot.png)

### Test Data Root Mean Squared Error

![](1.PNG)

### New Sample Data Prediction

![](2.PNG)

## RESULT
Thus a basic neural network regression model for the given dataset is written and executed successfully.
