import pandas as pd
import numpy as np


def convert_data():
    test = np.array(pd.read_csv('./data/fashion-mnist_test.csv'))
    train = np.array(pd.read_csv('./data/fashion-mnist_train.csv'))

    np.savez_compressed('./data/fashion-mnist.npz', train=train, test=test)

if __name__ == '__main__':
    print('Converting source data')
    convert_data()
    print('Conversion completed')