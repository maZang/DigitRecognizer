import sys
import pandas as pd
import numpy as np
import time
from sklearn import svm

def main():

    #Timing data read time
    startTime = time.time()

    # The competition datafiles are in the directory ../input
    # Read competition data files:
    train = pd.read_csv("./input/train.csv")
    test  = pd.read_csv("./input/test.csv")
    endTime = time.time() - startTime
    print(("---{} seconds to read data---").format(endTime))

    #Separate training examples from labels
    X_train =  train.values[:, 1:].astype(float)
    y = train.values[:, 0].astype(float)

    #Timing SVM creation
    startTime = time.time()

    #create a SVM with linear kernel
    linearSVM = svm.LinearSVC()
    linearSVM.fit(X_train,y)
    endTime = time.time() - startTime
    print(("---{} seconds to create SVM---").format(endTime))

    #Test data
    X_test = test.values.astype(float)


    #predict training set digits
    predictions = linearSVM.predict(X_train)

    print(predictions)
    print(y)

    #accuracy test
    correct = 0.0
    for index in range(y.shape[0]):
        if y[index] == predictions[index]:
            correct+=1
    print(("Accuracy: {}").format(correct/y.shape[0]))

    #test predictions
    return linearSVM.predict(X_test)


if __name__ == '__main__':
	main()