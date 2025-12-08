# Building Logistic Regression from Scratch
# Formula: y = σ(wx + b)
    # y: predicted output
    # σ: sigmoid function
    # w: weight (slope)
    # x: input feature
    # b: bias (intercept)
    
# Gradient Descent
# w = w - learning_rate * dw
# b = b - learning_rate * db

# Importing necessary libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score



class Logistic_Regression():

    # Declaring the hyperparameters (learning rate & epochs)
  
    def __init__(self, learning_rate, epochs):

        self.learning_rate = learning_rate
        self.epochs = epochs


    def fit(self, X, Y):

        # Number of training examples & number of features
        # Number of data points in the dataset (number of rows)  -->  m
        # Number of input features in the dataset (number of columns)  --> n
        self.m, self.n = X.shape 

        # Initiating the weight and bias 

        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # Implementing Gradient Descent
    
        for i in range(self.epochs):
            self.model_parameters()

    def model_parameters(self):
    
        # Sigmoid function implementation
    
        Y_hat =  1 / (1 + np.exp( - (self.X.dot(self.w) + self.b ) )) 
    
        dw = (1 / self.m) * np.dot(self.X.T, (Y_hat - self.Y))

        db = (1 / self.m) * np.sum(Y_hat - self.Y)

        # Updating the weights & bias
    
        self.w = self.w - self.learning_rate * dw
    
        self.b = self.b - self.learning_rate * db


    def predict(self, X):
    
        Y_prediction = 1 / (1 + np.exp( - (X.dot(self.w) + self.b ) )) 
        Y_prediction = np.where(Y_prediction > 0.5, 1, 0)
        return Y_prediction
    

# Data Loading & Preprocessing

df = pd.read_csv('diabetes.csv')
df.head()
df.shape
df.info()
df.describe()

# Checking for missing values

df.isnull().sum()

# Splitting the dataset into input features & target variable

X = df.drop('Outcome', axis=1).values
Y = df['Outcome'].values

df.groupby("Outcome").mean()

# 0 --> Non-Diabetic
# 1 --> Diabetic

# Standardizing the input features

scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X)

# Splitting the dataset into training & testing sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X.shape, X_train.shape, X_test.shape)

# Training the Logistic Regression model

model = Logistic_Regression(learning_rate=0.01, epochs=1000)
model.fit(X_train, Y_train)

# Making predictions on the dataset

X_train_pred = model.predict(X_train)
Y_train_pred = model.predict(X_test)

# Calculating the accuracy on the train set

train_accuracy = accuracy_score(Y_train, X_train_pred)
print(f'Training Accuracy: {train_accuracy * 100:.2f}%')

# Calculating the accuracy on the test set

test_accuracy = accuracy_score(Y_test, Y_train_pred)
print(f'Testing Accuracy: {test_accuracy * 100:.2f}%')

# Making predictions on a new data point

new_data = (7,124,67,15,200,13.2,0.673,46)
new_data = scaler.transform([new_data])
prediction = model.predict(new_data)

print(prediction)

if prediction[0] == 0:
    print("The person is Non-Diabetic")
else:
    print("The person is Diabetic")