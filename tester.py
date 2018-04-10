import tensorflow as tf
from trainer.datasource import DataSource



if __name__ == '__main__':

    model = tf.keras.models.load_model('example_model.h5')
    data = DataSource('./data/fashion-mnist.npz')

    print(model.summary())

    print(model.evaluate(data.X_test, data.Y_test, batch_size=32))