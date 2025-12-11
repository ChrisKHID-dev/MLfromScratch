# Building Support Vector Machine (SVM) Classifier from Scratch
# Formula: f(x) = wx + b
    # f(x): decision function
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

class SVM_Classifier():

    # Declaring the hyperparameters (learning rate, epochs & regularization parameter)
  
    def __init__(self, learning_rate, epochs, C):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.C = C


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

    
        for i in range(self.epochs):
            self.model_parameters()

    def model_parameters(self):
        
        # Label encoding: Converting 0 to -1 for SVM
        
        self.Y = np.where(self.Y <= 0, -1, 1)
    
        # Implementing Gradient Descent (Hinge Loss)
        
        for idx, x_i in enumerate(self.X):
            condition = self.Y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
            if (condition == True):
                dw = 2 * self.C * self.w
                db = 0
            else:
                dw = 2 * self.C * self.w - np.dot(x_i, self.Y[idx])
                db = self.Y[idx]
            
            # Updating weights and bias
            
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
    
    def predict(self, X):
        output = np.dot(X, self.w) - self.b
        pred_label = np.sign(output)
        y_hat =  np.where(pred_label <= 0, 0, 1)
        return y_hat
    

# Data Loading & Preprocessing

df = pd.read_csv("diabetes.csv")
df.head()
df.shape
df.describe()

df["Outcome"].value_counts()

# Non-Diabetic: 0
# Diabetic: 1

# Splitting the data into features & target variable
X = df.drop("Outcome", axis=1).values
Y = df["Outcome"].values
print(X.shape)
print(Y.shape)

# Data Standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X)

# Splitting the dataset into training & testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X.shape, X_train.shape, X_test.shape)

# Model Training
classifier = SVM_Classifier(learning_rate=0.001, epochs=1000, C=0.01)
classifier.fit(X_train, Y_train)

# Model Evaluation

# Calculating accuracy on training data
X_train_pred = classifier.predict(X_train)
X_train_accuracy = accuracy_score(Y_train, X_train_pred)
print("Training Accuracy:", X_train_accuracy)

# Calculating accuracy on testing data
X_test_pred = classifier.predict(X_test)
X_test_accuracy = accuracy_score(Y_test, X_test_pred)
print("Testing Accuracy:", X_test_accuracy)

# Building the Prediction System

def predict_system(input_data):
    
    # Changing the input data to numpy array
    input_data_as_numpy_array = np.array(input_data)

    # Reshaping the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Standardizing the input data
    std_data = scaler.transform(input_data_reshaped)

    # Making prediction
    prediction = classifier.predict(std_data)
    return prediction

# Example input data point
input_data = (4,110,92,0,0,37.6,0.191,30)

prediction = predict_system(input_data)
if (prediction[0] == 0):
    print("The person is Non-Diabetic")
else:
    print("The person is Diabetic")
