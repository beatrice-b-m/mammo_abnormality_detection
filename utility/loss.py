import tensorflow.keras.backend as K
"""
Loss Code Created on Mon Apr  5 15:01:37 2021
@author:  Asma Baccouche
"""


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def seg_loss(y_true, y_pred):
    return -(0.4*dice_coef(y_true, y_pred) + 0.6*iou_coef(y_true, y_pred))
