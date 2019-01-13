import sys
import pandas as pd

from models import *
from keras.utils.generic_utils import get_custom_objects
from keras import Model
from keras.layers import *

import seaborn as sns
import datetime
sns.set_style("white")

from skimage.transform import resize

from keras.preprocessing.image import load_img
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils.generic_utils import get_custom_objects
from keras.models import load_model

from losses import *
from metrics import *
from call_backs import *
from models import *

def main():
    ## fold desired to train on ##
    fold = sys.argv[0]
    model = sys.argv[1]

    MODEL_PATH = sys.argv[2]
    TRAIN_PATH = "https://github.com/GavinClarke0/TGS-Salt_Detection/blob/master/data/folds/trainData_fold" + fold + ".csv"
    VALIDATE_PATH = "https://github.com/GavinClarke0/TGS-Salt_Detection/blob/master/data/folds/testData_fold" + fold + ".csv"

    TARGET_SIZE = 128
    ORIGINAL_SIZE = 101

    def resizeUp(img):
        ## add check of image size
        return resize(img, (TARGET_SIZE, TARGET_SIZE, 1), mode='constant', preserve_range=True)

    def resizeDown(img):
        ## size down
        return resize(img, (ORIGINAL_SIZE, ORIGINAL_SIZE, 1), mode='constant', preserve_range=True)

    test_df = pd.read_csv(VALIDATE_PATH, index_col="id")
    train_df = pd.read_csv(TRAIN_PATH, index_col="id")

    test_df = test_df.drop(columns=['images', 'masks'])
    train_df = train_df.drop(columns=['images', 'masks'])

    train_df["images"] = [np.array(
        load_img("gdrive/My Drive/Kaggle/tgiSaltChallenge/train/images/{}.png".format(idx), grayscale=True)) / 255 for
                          idx in train_df.index]
    train_df["masks"] = [np.array(
        load_img("gdrive/My Drive/Kaggle/tgiSaltChallenge/train/masks/{}.png".format(idx), grayscale=True)) / 255 for
                         idx in train_df.index]

    test_df["images"] = [np.array(
        load_img("gdrive/My Drive/Kaggle/tgiSaltChallenge/train/images/{}.png".format(idx), grayscale=True)) / 255 for
                         idx in test_df.index]
    test_df["masks"] = [np.array(
        load_img("gdrive/My Drive/Kaggle/tgiSaltChallenge/train/masks/{}.png".format(idx), grayscale=True)) / 255 for
                        idx in test_df.index]

    train_imageData = np.array(train_df.images.map(resizeUp).tolist()).reshape(-1, TARGET_SIZE, TARGET_SIZE, 1)
    train_maskData = np.array(train_df.masks.map(resizeUp).tolist()).reshape(-1, TARGET_SIZE, TARGET_SIZE, 1)

    test_imageData = np.array(test_df.images.map(resizeUp).tolist()).reshape(-1, TARGET_SIZE, TARGET_SIZE, 1)
    test_maskData = np.array(test_df.masks.map(resizeUp).tolist()).reshape(-1, TARGET_SIZE, TARGET_SIZE, 1)

    ## Augment data with reflection
    x_train = np.append(train_imageData, [np.fliplr(x) for x in train_imageData], axis=0)
    y_train = np.append(train_maskData, [np.fliplr(x) for x in train_maskData], axis=0)

    if (MODEL_PATH == ''):

        if (model == 'ResNet50_Unet'):

            model = Resnet50_Unet().get_unet_resnet((TARGET_SIZE, TARGET_SIZE, 1))
            model_Name = 'ResNet50_Unet'

        elif (model == "VGG16_Unet"):

            model = Vgg16_Unet((TARGET_SIZE, TARGET_SIZE, 1)).return_Model()
            model_Name = "VGG16_Unet"

    else:
        model_Name = model

        model = load_model(MODEL_PATH)

        get_custom_objects().update({"bce_dice_loss": bce_dice_loss})
        get_custom_objects().update({"lovasz_loss": lovasz_loss})
        get_custom_objects().update({"my_iou_metric": my_iou_metric})

    model.compile(loss=lovasz_loss, optimizer="sgd", metrics=[my_iou_metric, Kaggle_IoU_Precision, "accuracy"])

    epochs = 50
    batch_size = 24

    swa = SWA('TGS-Salt_Detection/models/'+str(datetime.datetime.now().day)+"_"+model_Name+'.py', 47)

    ## PLay with early stopping etc ##c
    early_stopping = EarlyStopping(patience=50, verbose=1)
    ##model_checkpoint = ModelCheckpoint("gdrive/My Drive/Kaggle/tgiSaltChallenge/ModelCheckPoints/TestModel_ResNet50_Jan11_firstTrain.h5", save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.0000001, verbose=1)
    snapshot = SnapshotCallbackBuilder(nb_epochs=epochs, nb_snapshots=1, init_lr=0.01)

    model.fit(x_train, y_train,
                        validation_data=[test_imageData, test_maskData],
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=snapshot.get_callbacks())

if __name__ == '__main__':
    main()

