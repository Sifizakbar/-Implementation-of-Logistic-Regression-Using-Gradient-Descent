# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Read the dataset, convert the status column into 0 and 1, and select the required input features and output.
2.Scale the input features using standardization so all values are in a similar range.
3.Add a column of ones to the input data to include the bias term.
4.Initialize weights to zero and define the sigmoid function and cost function.
5.Repeat for fixed iterations: compute predictions, find error (gradient), update weights, and store cost.
6.Predict final outputs using threshold 0.5, calculate accuracy, and plot cost vs iterations

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Sifiz A
RegisterNumber:  212225040414

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("Placement_Data (1).csv")
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})
X = data[['ssc_p', 'mba_p']].values
y = data['status'].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
m = len(y)
X = np.c_[np.ones(m), X]
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def cost_function(X, y, theta):
    h = sigmoid(X @ theta)
    return (-1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))
theta = np.zeros(X.shape[1])
alpha = 0.12
cost_history = []
for i in range(500):
    z = X @ theta
    h = sigmoid(z)
    gradient = (1/m) * X.T @ (h - y)
    theta = theta - alpha * gradient
    cost = cost_function(X, y, theta)
    cost_history.append(cost)
y_pred = (sigmoid(X @ theta) >= 0.5).astype(int)
accuracy = np.mean(y_pred == y) * 100
print("Weights:", theta)
print("Accuracy:", accuracy, "%")
plt.figure()
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Logistic Regression using Gradient Descent")
plt.show()

```

## Output:
![logistic regression using gradient descent](sam.png)
<img width="1594" height="869" alt="Screenshot (446)" src="https://github.com/user-attachments/assets/ba7d077f-67b2-4d51-a297-3d1259d1f407" />
<img width="1685" height="866" alt="Screenshot (447)" src="https://github.com/user-attachments/assets/58500f8b-7160-40fe-ae2d-2d81d24b71c4" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

