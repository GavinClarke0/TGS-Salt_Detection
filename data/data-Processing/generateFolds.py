import pandas as pd
import numpy as np

from keras.preprocessing.image import load_img

from skimage.transform import resize
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

# Constants #
TARGET_SIZE = 128
ORIGINAL_SIZE = 101

## functions ##
def resizeUp(img):
    ## add check of image size
    return resize(img, (TARGET_SIZE, TARGET_SIZE, 1), mode='constant', preserve_range=True)

def resizeDown(img):
    ## size down
    return resize(img, (ORIGINAL_SIZE, ORIGINAL_SIZE, 1), mode='constant', preserve_range=True)

## stratify by sale coverage
def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i


# Data Loading and feature creation
train_df = pd.read_csv("train.csv", index_col="id", usecols=[0,1])
depths_df = pd.read_csv("depths.csv", index_col="id")
train_df = train_df.join(depths_df)

train_df["images"] = [np.array(load_img("train\images\{}.png".format(idx), grayscale=True, )) / 255 for idx in train_df.index]
train_df["masks"] = [np.array(load_img("train\masks\{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]
train_df["coverage"] = train_df.masks.map(np.sum) / pow(ORIGINAL_SIZE, 2)
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

test_df = depths_df[~depths_df.index.isin(train_df.index)]

test_df.to_csv("test_csv.csv")


## stratify by sale coverage

## K fold validation ##
K = 5

x_train = train_df.index
y_train = train_df.coverage_class


folds = list(StratifiedKFold(n_splits= K, shuffle=True, random_state=12).split(x_train, y_train))

for i in range(5):

    trainingdata = train_df.iloc[folds[0][0]]
    testdata = train_df.iloc[folds[0][1]]

    trainingdata.to_csv("C:\\Users\Gavin Clarke\Documents\Kaggle\SaltDepositID\\folds\\trainData_fold" + str(i))
    testdata.to_csv("C:\\Users\Gavin Clarke\Documents\Kaggle\SaltDepositID\\folds\\testData_fold" + str(i))
