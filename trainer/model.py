import tensorflow as tf

class Model:

    def __init__(self):
        NUMBER_CLASSES = 10
        INPUT_WIDTH = 28
        INPUT_HEIGHT = 28


        keras = tf.keras
        layers = tf.keras.layers

        model = keras.models.Sequential()
        model.add(layers.InputLayer(input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 1)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, 3, activation='relu'))
        model.add(layers.MaxPool2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.4))
        model.add(layers.BatchNormalization())
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(NUMBER_CLASSES, activation='softmax'))

        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model
