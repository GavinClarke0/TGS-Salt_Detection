import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

import tensorflow as tf

# keras
from keras.models import load_model
from keras.preprocessing.image import  load_img
from keras.layers import *
from keras.losses import binary_crossentropy
from keras.utils.generic_utils import get_custom_objects

from skimage.transform import resize
from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook

## FIXED VALUES ##
TARGET_SIZE = 192
ORIGINAL_SIZE = 101

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(dice_loss(y_true, y_pred))

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, K.sigmoid(y_pred)) + dice_loss(y_true, K.sigmoid(y_pred))

def reduceChannels(imgArray):
    return resize(imgArray, (ORIGINAL_SIZE, ORIGINAL_SIZE, 1), mode='constant', preserve_range=True)

def resizeUp(img):
    ## add check of image size
    return resize(img, (TARGET_SIZE, TARGET_SIZE), mode='constant', preserve_range=True)

def resizeDown(img):
    ## size down
    return resize(img, (ORIGINAL_SIZE, ORIGINAL_SIZE), mode='constant', preserve_range=True)

def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)

def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i

def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch] > 0, B[batch] > 0
        #         if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
        #             metric.append(0)
        #             continue
        #         if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
        #             metric.append(0)
        #             continue
        #         if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
        #             metric.append(1)
        #             continue

        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)


def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)


def my_iou_metric_2(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0], tf.float64)


# Data Loading and feature creation)
train_df = pd.read_csv("train.csv", index_col="id", usecols=[0, 1])
depths_df = pd.read_csv("depth.csv", index_col="id")
train_df["coverage"] = train_df.masks.map(np.sum) / pow(ORIGINAL_SIZE, 2)
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

test_df = depths_df[~depths_df.index.isin(train_df.index)]

# split data #

ids_train, ids_valid, x_train, x_valid, y_train, y_valid, depth_train, depth_test = train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(resizeUp).tolist()).reshape(-1, TARGET_SIZE, TARGET_SIZE, 1),
    np.array(train_df.masks.map(resizeUp).tolist()).reshape(-1, TARGET_SIZE, TARGET_SIZE, 1),
    train_df.z.values,
    test_size=0.1, stratify=train_df.coverage_class, random_state= 444)

get_custom_objects().update({"bce_dice_loss": bce_dice_loss})
get_custom_objects().update({"my_iou_metric": my_iou_metric})

## model loading
model = load_model("C:\\Users\Gavin Clarke\Downloads\TestModel_ResNet50_v1 (2).h5")

preds_valid = model.predict(x_valid).reshape(-1, TARGET_SIZE, TARGET_SIZE, 1)
preds_valid = np.array([reduceChannels(x) for x in preds_valid])
## preds_valid = np.array([reduceChannels(x) for x in preds_valid]) ##

y_valid_ori = np.array([train_df.loc[idx].masks for idx in ids_valid])

## review: thresholds
thresholds = np.linspace(0, 1, 50)
ious = np.array([iou_metric_batch(y_valid_ori, np.int32(preds_valid > threshold)) for threshold in tqdm_notebook(thresholds)])

## figure out why and how this works [
## ious[9:-10] is likey fucking trash ##
threshold_best_index = np.argmax(ious[9:-10]) + 9
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

# predicts & scores
x_test = np.array([resizeUp(np.array(load_img("test.csv".format(idx), grayscale=True))) / 255 for idx in
         tqdm_notebook(test_df.index)]).reshape(-1, TARGET_SIZE, TARGET_SIZE, 1)

preds_test = model.predict(x_test)
preds_test = np.array([reduceChannels(x) for x in preds_test])

pred_dict = {idx: rle_encode(np.round(reduceChannels(preds_test[i]) > threshold_best)) for i, idx in enumerate(tqdm_notebook(test_df.index.values))}
sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
