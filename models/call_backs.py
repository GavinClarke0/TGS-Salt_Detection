import keras
import keras.callbacks as callbacks
from keras.layers import *

import seaborn as sns
sns.set_style("white")

""" https://www.kaggle.com/meaninglesslives/getting-0-87-on-private-lb-using-kaggle-kernel
"""

## Stochastic weight averaging ## 

class SWA(keras.callbacks.Callback):

    def __init__(self, filepath, swa_epoch):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch

    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print('Stochastic weight averaging selected for last {} epochs.'
              .format(self.nb_epoch - self.swa_epoch))

    def on_epoch_end(self, epoch, logs=None):

        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()

        elif epoch > self.swa_epoch:
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = (self.swa_weights[i] *
                                       (epoch - self.swa_epoch) + self.model.get_weights()[i]) / (
                                                  (epoch - self.swa_epoch) + 1)

        else:
            pass

    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        self.model.save(self.filepath)
        ## self.model.save_weights(self.filepath+"_weights")
        print('Final stochastic averaged weights saved to file.')


class SnapshotCallbackBuilder:
    def __init__(self, nb_epochs, nb_snapshots, swa, init_lr=0.1, ):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr
        self.swa = swa

    def get_callbacks(self, model_prefix='Model'):
        callback_list = [
            callbacks.ModelCheckpoint(
                "gdrive/My Drive/Kaggle/tgiSaltChallenge/ModelCheckPoints/TestModel_ResNet50_Jan11_firstTrain_checkpoint.h5",
                monitor='val_my_iou_metric',
                mode='max', save_best_only=True, verbose=1),
            self.swa,
            callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule),
            callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0000001, verbose=1)
        ]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)
