import tensorflow.keras.backend as K
import tensorflow as tf

def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def weighted_binary_crossentropy(y_true, y_pred):
    y_true = K.cast(y_true, 'float')
    bce = K.binary_crossentropy(y_true, y_pred)

    one_weights = [0.94, 0.78]
    zero_weights = [1.07, 1.4]

    weight_vector = y_true * one_weights + (1. - y_true) * zero_weights
    weighted_bce = weight_vector * bce

    return K.mean(weighted_bce)

def f1_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float')

    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def dice_loss(y_true, y_pred, smooth = 1):
    y_true = K.cast(y_true, 'float')

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - dice_coef

def DiceBCE_Loss(targets, inputs, smooth = 1e-6):    
    targets = K.cast(targets, 'float')
    
    BCE = K.binary_crossentropy(targets, inputs)  
    dice = dice_loss(targets, inputs)
    Dice_BCE = BCE + dice
    
    return Dice_BCE

def DiceWBCE_Loss(targets, inputs, smooth = 1e-6):    
    targets = K.cast(targets, 'float')
    
    BCE = weighted_binary_crossentropy(targets, inputs)  
    dice = dice_loss(targets, inputs)
    Dice_BCE = BCE + dice
    
    return Dice_BCE

def IoU_Loss(targets, inputs, smooth=1e-6):
    targets = K.cast(targets, 'float')
    
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    intersection = K.sum(K.dot(targets, inputs))
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU