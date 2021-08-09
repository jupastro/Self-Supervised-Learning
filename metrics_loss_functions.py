import tensorflow as tf

def jaccard_index( y_true, y_pred, skip_first_mask=False ):
    ''' Define Jaccard index for multiple labels.
        Args:
            y_true (tensor): ground truth masks.
            y_pred (tensor): predicted masks.
            skip_background (bool, optional): skip 0-label from calculation.
        Return:
            jac (tensor): Jaccard index value
    '''
    t=0.5
    if tf.shape(y_true)[-1]==1:
          y_pred_ = tf.cast(y_pred>t , dtype=tf.int32)
          y_true = tf.cast(y_true, dtype=tf.int32)

          TP = tf.math.count_nonzero(y_pred_ * y_true)
          FP = tf.math.count_nonzero(y_pred_ * (y_true - 1))
          FN = tf.math.count_nonzero((y_pred_ - 1) * y_true)

          jac = tf.cond(tf.greater((TP + FP + FN), 0), lambda: TP / (TP + FP + FN), lambda: tf.cast(0.000, dtype='float64'))
    else:
        # We read the number of classes from the last dimension of the true labels
        num_classes = tf.shape(y_true)[-1]
        # One_hot representation of predicted segmentation after argmax
        y_pred_ = tf.one_hot(tf.math.argmax(y_pred, axis=-1), num_classes)
        y_pred_ = tf.cast(y_pred_, dtype=tf.int32)
        # y_true is already one-hot encoded
        y_true_ = tf.cast(y_true, dtype=tf.int32)
        # Skip background pixels from the Jaccard index calculation
        if skip_first_mask:
          y_true_ = y_true_[...,1:]
          y_pred_ = y_pred_[...,1:]

        TP = tf.math.count_nonzero(y_pred_ * y_true_)
        FP = tf.math.count_nonzero(y_pred_ * (y_true_ - 1))
        FN = tf.math.count_nonzero((y_pred_ - 1) * y_true_)

        jac = tf.cond(tf.greater((TP + FP + FN), 0), lambda: TP / (TP + FP + FN),
                      lambda: tf.cast(0.000, dtype='float64'))

    return jac

def jaccard_index_final(y_true, y_pred, t=0.5):
  """Define Jaccard index for final evaluation .
      Args:
          y_true (tensor): ground truth masks.
          y_pred (tensor): predicted masks.
          t (float, optional): threshold to be applied.
      Return:
          jac (tensor): Jaccard index value
      additional: this metric is meant to output the same as the jaccard_index above but only for a single mask
  """

  y_pred_ = tf.cast(y_pred>t , dtype=tf.int32)
  y_true = tf.cast(y_true>0, dtype=tf.int32)

  TP = tf.math.count_nonzero(y_pred_ * y_true)
  FP = tf.math.count_nonzero(y_pred_ * (y_true - 1))
  FN = tf.math.count_nonzero((y_pred_ - 1) * y_true)

  jac = tf.cond(tf.greater((TP + FP + FN), 0), lambda: TP / (TP + FP + FN),
                lambda: tf.cast(0.000, dtype='float64'))

  return jac

from tensorflow.keras import losses

def dice_coeff(y_true, y_pred):
    """Define Dice coefficient.
       Args:
            y_true (tensor): ground truth masks.
            y_pred (tensor): predicted masks.
       Return:
            score (tensor): Dice coefficient value
    """
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

# Dice coefficient loss (1 - Dice coefficient)
def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

# Loss function combining binary cross entropy and Dice loss
def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

# Weighted BCE+Dice
# Inspired by https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0
def weighted_bce_dice_loss(w_dice=0.5, w_bce=0.5):
    def loss(y_true, y_pred):
        return losses.binary_crossentropy(y_true, y_pred) * w_bce + dice_loss(y_true, y_pred) * w_dice
    return loss




import keras.backend as K  
#Based in denoiseg loss function 
def loss_seg(relative_weights=[1.0,1.0,5.0]):
    """
    It is based in the DenoiSeg training function used in their paper for segmentation
    Calculates Cross-Entropy Loss between the class targets and predicted outputs.
    Predicted outputs consist of three classes: Foreground, Background and Border.
    Class predictions are weighted by the parameter `relative_weights`.
    
    """

    class_weights = tf.constant([relative_weights])
    def seg_crossentropy(class_targets, y_pred):
  
      
        onehot_labels = tf.reshape(class_targets, [-1, 3])# maintains the 3 dimensions for the labels regardless the size of the image or the batch size
        weights = tf.reduce_sum(class_weights * onehot_labels, axis=1)#performs a weighted sum over a particular dimension of the tensor

        a = tf.reduce_sum(onehot_labels, axis=-1)#performs once again a sum over a particular dimension

        loss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels,
                                                          logits=tf.reshape(y_pred, [-1, 3]))#computes the softmax cat crossentropy

        weighted_loss = loss * weights #obtains a loss weighted by the number of labels and samples (the more positive of a label the higher importance )

        return K.mean(a * weighted_loss)# weights once again the number of positive labels per class
    return seg_crossentropy
        
