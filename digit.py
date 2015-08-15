import pandas as pd
import numpy as np
import math
from scipy import optimize
from scipy.sparse import *

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
    lamda = 0.085
    theta = oneVsAll(X_trainer, y, num_classes, lamda);
    predict = predicter(theta, X_tester)
    imgid = np.zeros((predict.shape[0],1))
    for n in range(0, predict.shape[0]):
        imgid[n] = n + 1
    predict = np.c_[imgid, predict]
    np.savetxt('predict.csv', predict, fmt="%i", delimiter= ",", header = "ImageId,Label", comments = "")

def sigmoid(X):
    sig = 1.0 + math.e ** (-1.0 * X)
    sig = 1.0 / sig
    sig[sig == 0] = 1.0 / (10 ** 20)
    return sig


def oneVsAll(X_trainer, y, num_classes, lamda):
    (m, n) = X_trainer.shape
    #initialize theta to be empty
    full_theta = np.empty(shape=[num_classes, n + 1])
    #add column of 1's to X
    X_trainer = np.c_[np.ones((m,1)), X_trainer]
    for num in range(0, num_classes):
        print("Training class number " + str(num))
        initial_theta = np.zeros((n + 1, 1))
        y_num = find_vector(y, num)
        args = (X_trainer, y_num, lamda, m, n)
        #cost(initial_theta, X_trainer, y_num, lamda, m, n)
        #grad(initial_theta, X_trainer, y_num, lamda, m, n)
        theta = optimize.fmin_cg(cost, initial_theta, fprime = grad, args = args, maxiter = 100)
        full_theta[num] = theta
    print(full_theta)
    return full_theta

def find_vector(y, num):
    m = y.shape[0]
    y_num = np.zeros((m,1))
    for n,i in enumerate(y):
        if i == num:
            y_num[n] = 1
    return y_num

def cost(theta, *args):
    X_trainer, y, lamda, m, n = args
    theta = theta.reshape((n+1, 1))
    X_sparse = csc_matrix(X_trainer)
    h = sigmoid(X_sparse * theta)
    #print("Calculating cost...")
    logh = np.log(h).reshape((m, 1))
    onelogh = np.log(1-h).reshape((m, 1))
    J = 1/m * (-1 * np.multiply(y, logh) - np.multiply((1 - y), onelogh)).sum() + lamda/(2 * m) * ((theta ** 2).sum() - theta[0] ** 2)
    #print(J)
    return J

def grad(theta, *args):
    X_trainer, y, lamda, m, n = args
    theta = theta.reshape((n+1, 1))
    a = np.ones((n+1, 1))
    a[0] = 0
    X_sparse = csc_matrix(X_trainer)
    h = sigmoid(X_sparse * theta)
    error = h.reshape((m,1)) - y
    X_sparse = csc_matrix(X_trainer.T)
    #print("Calculating gradient...")
    grad = 1/m * X_sparse * error +lamda/m * np.multiply(theta, a)
    grad = grad.flatten()
    return grad

def predicter(theta, X_tester):
    m = X_tester.shape[0]
    X_tester = np.c_[np.ones((m,1)), X_tester]
    X_sparse = csc_matrix(X_tester)
    y_test = X_sparse * theta.T
    predict = y_test.argmax(1)
    return predict

main()