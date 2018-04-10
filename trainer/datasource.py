import numpy as np
from io import BytesIO
from tensorflow.python.lib.io import file_io
from tensorflow.python.keras.utils import to_categorical


class DataSource:

    def __init__(self, source):
        data_stream = BytesIO(file_io.read_file_to_string(source, binary_mode=True))
        data = np.load(data_stream)
        test = data['test']
        train = data['train']

        np.random.shuffle(train)
        X_train = train[:, 1:].astype(np.float32).reshape(-1, 28, 28, 1)
        X_train /= 255
        Y_train = to_categorical(train[:, 0].astype(np.float32))

        X_test = test[:, 1:].astype(np.float32).reshape(-1, 28, 28, 1)
        X_test /= 255
        Y_test = to_categorical(test[:, 0].astype(np.float32))

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
