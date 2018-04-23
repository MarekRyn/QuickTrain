import tensorflow as tf
import argparse

from trainer.model import Model
from trainer.datasource import DataSource
from trainer.saver import Saver


if __name__ == '__main__':

    MAX_EPOCHS = 100
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.1
    SAVE_BEST_ONLY = True

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-source', help='GCS or local path to training data', required=True)
    parser.add_argument('--job-dir', help='GCS or local path to write checkpoints and export models', required=True)
    parser.add_argument('--log-dir', help='GCS or local path to write logs', required=True)
    parser.add_argument('--verbose', help='Verbose level', required=False, default=1)
    args = parser.parse_args()
    arguments = args.__dict__

    # Validating and compiling model
    model = Model().model

    # Loading training and test data
    data = DataSource(arguments['data_source'])

    # Initializing saver for trained models
    saver = Saver(model, arguments['job_dir'], best_only=SAVE_BEST_ONLY)

    # Initializing callbacks
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=arguments['log_dir']),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=0, mode='auto'),
        tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: saver.save(epoch, logs))
    ]

    # Training of model
    model.fit(data.X_train, data.Y_train,
              epochs=MAX_EPOCHS,
              batch_size=BATCH_SIZE,
              verbose=int(arguments['verbose']),
              validation_data=(data.X_test, data.Y_test),
              callbacks=callbacks)
