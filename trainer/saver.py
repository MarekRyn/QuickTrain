from tensorflow.python.lib.io import file_io
from os import path


class Saver:

    def __init__(self, model, job_dir, prefix='model', best_only=False, min_acc=0, max_loss=100):
        self.model = model
        self.job_dir = job_dir
        self.prefix = prefix
        self.best_only = best_only
        self.min_acc = min_acc
        self.max_loss = self._loss = max_loss

    def save(self, epoch, logs):
        loss = logs.get('val_loss', logs.get('loss', 100))
        acc = logs.get('val_acc', logs.get('acc', 0))
        try:
            assert loss < self.max_loss
            assert acc > self.min_acc

            if self.best_only:
                assert loss < self._loss

            self._loss = loss

            epoch = str(epoch).zfill(4)
            loss = str(round(loss, 2)).zfill(2)
            acc = str(round(acc, 2)).zfill(2)
            name = "{}_{}_{}_{}.h5".format(self.prefix, epoch, acc, loss)

            self.model.save(name)
            with file_io.FileIO(name, mode='rb') as input_f:
                with file_io.FileIO(path.join(self.job_dir, name), mode='wb+') as output_f:
                    output_f.write(input_f.read())
        except AssertionError:
            pass

