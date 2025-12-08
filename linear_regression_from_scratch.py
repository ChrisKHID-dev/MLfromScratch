# Building Linear Regression from Scratch
# Formula: y = wx + b
    # y: predicted output
    # w: weight (slope)
    # x: input feature
    # b: bias (intercept)

# Gradient Descent 
# w = w - learning_rate * dw
# b = b - learning_rate * db

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Linear_Regression():

  # Declaring the hyperparameters (learning rate & epochs)
  
  def __init__(self, learning_rate, epochs):

    self.learning_rate = learning_rate
    self.epochs = epochs


  def fit(self, X, Y):

    # Number of training examples & number of features

    self.m, self.n = X.shape  # number of rows & columns

    # Initiating the weight and bias 

    self.w = np.zeros(self.n)
    self.b = 0
    self.X = X
    self.Y = Y

    # Implementing Gradient Descent
    
    for i in range(self.epochs):
      self.model_parameters()


  def model_parameters(self):

    Y_prediction = self.predict(self.X)

    # Calculate gradients

    dw = - (2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m

    db = - 2 * np.sum(self.Y - Y_prediction)/self.m

    # Updating the model parameters
    
    self.w = self.w - self.epochs * dw
    self.b = self.b - self.epochs * db
 

  def predict(self, X):

    return X.dot(self.w) + self.b

# Data Pre-processing

# Loading the dataset

salary_data = pd.read_csv('salary_data.csv')
print(salary_data.head())
print(salary_data.info())
print(salary_data.describe())


# Checking for missing values

print(salary_data.isnull().sum())

# Splitting the dataset into input features and target variable

X = salary_data.drop('Salary', axis=1).values
print(X)
Y = salary_data['Salary'].values
print(Y)

# Splitting the dataset into training and testing sets

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Training the Linear Regression model

model = Linear_Regression(learning_rate=0.01, epochs=100)
model.fit(X_train, Y_train)

# Printing the model parameters

print("Weight:", model.w[0])
print("Bias:", model.b)

# Predicting the test set results

Y_pred = model.predict(X_test)
print("Predicted values:", Y_pred)

# Visualizing the results

plt.scatter(X_test, Y_test, color='red', label='Actual Data')
plt.scatter(X_test, Y_pred, color='blue', label='Predicted Data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression: Salary vs Experience')
plt.legend()
plt.show()

# Evaluating the model using Mean Absolute Error (MAE)

MAE = np.mean(np.abs(Y_test - Y_pred))
print("Mean Absolute Error:", MAE)