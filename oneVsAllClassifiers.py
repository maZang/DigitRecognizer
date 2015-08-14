import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier as oneVsAll

def main():
    # The competition datafiles are in the directory ../input
    # Read competition data files:
    train = pd.read_csv("./input/train.csv")
    test  = pd.read_csv("./input/test.csv")

    #Separate training examples from labels
    X_train =  train.values[:, 1:].astype(float)
    y = train.values[:, 0].astype(float)

    


if __name__ == '__main__':
	main()