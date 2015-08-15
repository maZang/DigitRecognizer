import sys
import pprint
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

    #variety of C tried, none made significant difference, even after overfitting training data, default constructer used
    linearSVM = svm.LinearSVC()
    linearSVM.fit(X_train,y)
    endTime = time.time() - startTime
    print(("---{} seconds to create SVM---").format(endTime))

    #Test data
    X_test = test.values.astype(float)


    #predict training set digits
    predictions = linearSVM.predict(X_train).astype(int)

    print(("Predictions: {}").format(predictions))
    print(("Actual: {}").format(y.astype(int)))




    #algorithm analysis, debugging, and other info
    correct = 0.0
    numberOfDigits = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    numberOfDigitsWrong = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    #iterate through all examples
    for index in range(y.shape[0]):
        if y[index] == predictions[index]:
            correct+=1
        else:
            numberOfDigitsWrong[int(y[index])]+=1
        numberOfDigits[int(y[index])]+=1
        
    #results of analysis
    print(("Accuracy of algorithm on training data: {}").format(correct/y.shape[0]))
    print(("Number of each digit: {}").format(numberOfDigits))
    print(("Number of each digit predicted incorrectly: {}").format(numberOfDigitsWrong))





    #test predictions
    predictTest = (linearSVM.predict(X_test)).astype(int)
    imgid = np.zeros((predictTest.shape[0],1))
    for n in range(0, predictTest.shape[0]):
        imgid[n] = n + 1
    output = np.c_[imgid, predictTest]

    np.savetxt('output.csv', output, fmt="%i", delimiter= ",", header = "ImageId,Label", comments = "")
    return predictTest


if __name__ == '__main__':
    main()
