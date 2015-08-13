import pandas as pd
import numpy as np
from scipy import optimize

def main():
    # The competition datafiles are in the directory ../input
    # Read competition data files:
    train = pd.read_csv("./input/train.csv")
    test  = pd.read_csv("./input/test.csv")
    
    #ten classes for this problem
    num_classes = 10
    
    #Create the matrices
    X_trainer = train.values[:, 1:].astype(float)
    X_tester = test.values.astype(float)
    y = train.values[:, 0]

    #Train multiclass logistic regression
    print("Training multiclass logistic regression...")
    lamda = 0.1
    theta = oneVsAll(X_trainer, y, num_classes, lamda);

def sigmoid(X):
    sig = 1.0 + np.exp(-1.0 * X)
    sig = 1.0 / sig
    return sig


def oneVsAll(X_trainer, y, num_classes, lamda):
    (m, n) = X_trainer.shape
    #initialize theta to be empty
    full_theta = np.empty(shape=[n + 1, num_classes])
    #add column of 1's to X
    X_trainer = np.c_[np.ones((m,1)), X_trainer]
    for num in range(1, num_classes + 1):
        print("Training class number " + str(num))
        initial_theta = np.zeros((n + 1, 1))
        y_num = find_vector(y, num)
        args = (X_trainer, y_num, lamda, m, n)
        print("Starting optimization...")
        theta = optimize.fmin_cg(cost, initial_theta, fprime = grad, args = args)
    print(full_theta)

def find_vector(y, num):
    m = y.shape[0]
    y_num = np.zeros((m,1))
    for n,i in enumerate(y):
        if i == num:
            y_num[n] = 1
    return y_num

def cost(theta, *args):
    X_trainer, y, lamda, m, n = args
    J = 1/m * (-1 * np.multiply(y, np.log(sigmoid(np.dot(X_trainer , theta)))) - np.multiply((1 - y), np.log(1-sigmoid(np.dot(X_trainer , theta))))).sum() + lamda/(2 * m) * ((theta ** 2).sum() - theta[0] ** 2)
    return J

def grad(theta, *args):
    X_trainer, y, lamda, m, n = args
    a = np.ones((n, 1))
    a[0] = 0
    grad = 1/m * np.dot(X_trainer.T, sigmoid(np.dot(X_trainer, theta) - y)) + lamda/m * np.multiply(theta, a)
    return grad

main()