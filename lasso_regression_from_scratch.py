# Building Lasso Regression from Scratch
# Formula: y = wx + b
    # y: predicted output
    # w: weight (slope)
    # x: input feature
    # b: bias (intercept)   
# Lasso Regression adds L1 regularization to Linear Regression
# Cost Function with L1 regularization:
# J(w, b) = (1/m) * Σ(y_i - (wx_i + b))^2 + λ * Σ|w_j|
# where λ is the regularization parameter

# Gradient Descent with L1 regularization
# w = w - learning_rate * (dw + λ * sign(w))
# b = b - learning_rate * db

# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score

class Lasso_Regression():

    # Declaring the hyperparameters (learning rate, epochs & regularization parameter)
  
    def __init__(self, learning_rate, epochs, lambda_param):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_param = lambda_param


    def fit(self, X, Y):

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

        # Linear equation 
        Y_prediction = self.predict(self.X)

        # Calculate gradients
        dw = np.zeros(self.n)

        for i in range(self.n):

            if self.w[i]>0:
                dw[i] = (-(2*(self.X[:,i]).dot(self.Y - Y_prediction)) + self.lambda_param) / self.m
            else: 
                dw[i] = (-(2*(self.X[:,i]).dot(self.Y - Y_prediction)) - self.lambda_param) / self.m

        db = - 2 * np.sum(self.Y - Y_prediction)/self.m

        # Updating the model parameters
    
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db
    
    def predict(self, X):
        return X.dot(self.w) + self.b


# Loading the dataset
df = pd.read_csv("salary_data.csv")
df.head()
df.shape   
df.isnull().sum()

# Splitting the dataset into input features and target variable
X = df.drop("Salary", axis=1).values
Y = df["Salary"].values
print(X)
print(Y)

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Training the Lasso Regression model from scratch
model = Lasso_Regression(learning_rate=0.01, epochs=1000, lambda_param=0.1)
model.fit(X_train, Y_train)

# Making predictions on the test set
Y_pred = model.predict(X_test)
print("Predictions from Scratch:", Y_pred)


# Model Evaluation

# Mean absolute error
mae = mean_absolute_error(Y_test, Y_pred)
print("Mean Absolute Error from Scratch:", mae)

# R-squared score
r2 = r2_score(Y_test, Y_pred)
print("R-squared Score from Scratch:", r2)

# Comparing with errors with sklearn Lasso Regression
sk_lasso = Lasso()
sk_lasso.fit(X_train, Y_train)
Y_sk_pred = sk_lasso.predict(X_test)

# Mean absolute error
sk_mae = mean_absolute_error(Y_test, Y_sk_pred)
print("Mean Absolute Error from Sklearn:", sk_mae)

# R-squared score
sk_r2 = r2_score(Y_test, Y_sk_pred)
print("R-squared Score from Sklearn:", sk_r2)
    