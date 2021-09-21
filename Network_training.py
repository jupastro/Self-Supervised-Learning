#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:06:33 2021

@author: juliopastor
"""


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from SelfSupervisedLearning.metrics_loss_functions import jaccard_index,jaccard_index_final, dice_coeff, dice_loss ,bce_dice_loss,weighted_bce_dice_loss,loss_seg
from SelfSupervisedLearning.oneCycle import OneCycleScheduler
from SelfSupervisedLearning.NetworkDefinitions import UNet,ResUNet,MobileNetEncoder
from SelfSupervisedLearning.Data_aug import get_train_val_generators,random_90rotation
import tensorflow as tf
import numpy as np
def train(X_train,Y_train,X_val,Y_val,numEpochs,output_channels,patience,lr,min_lr,batch_size_value,schedule,model_name,optimizer_name,loss_acronym,max_pooling,train_encoder=True,random_encoder_weights=True,preTrain=False,Denoising=False,pre_load_weights=False,pretrained_model=None,plot_history=False,seg_weights=[1.,1.,5.],bottleneck_freezing=False,save_best_only=True):
  """Inputs:
        --------------------
        DATA:
        X_train(tensor):contaning the training patches to be augmented 
        Y_train(tensor):contaning the training labels to be augmented 
        X_val(tensor):contaning the validation patches to be augmented 
        Y_val(tensor):contaning the validation labels to be augmented 
        ---------------------
        HYPERPARAMS:
        numEpochs(int):number of "loops" of training
        output_channels(int):number of predictions to be performed by the last activation layer ( if it's bigger than 1 a softmax activation is used instead of a sigmoid)
        patience(int): number of "loops" without improvement till the training is stopped, in the case of reduce till the lr is reduced to its half
        lr(float): number indicating the lr starting value// in the case of oneCycle the max lr
        min_lr(float):minimum lr at which the schedulers should stop reducing it
        batch_size_value(int):number of images in each step of training inside an epoch
        schedule(string):indicating the variations performed in the lr during the training #'oneCycle' # 'reduce' # None
        model_name(string):indicating the architecture to be used #'UNet','ResUNet','MobileNetEncoder','AttentionNet'
        loss_acronym(string): indicating the name of the loss function to be applied 'BCE', 'Dice', 'W_BCE_Dice','CCE','SEG','mae','mse'
        optimizer_name(string):indicating the kind of optimized to be used 'Adam', 'SGD'
        max_pooling(boolean):indicating True if max_pooling must be performed, False if average pooling has to be performed
        preTrain(boolean):indicating whether we're preTraining with denoising the network or training it for the final task
        train_encoder(boolean):indicating whether to freeze or not the training of the encoder part of the model in the case of MobileNetEncoder
        random_encoder_weights(boolean):indicating whether to initialize the model with random_gaussian weights or to use MobileNet imagenet pretrained weights//only available for MobileNet encoder
        Denoising(boolean):whether to tune the architecture of the network(by varying its last layer to be able to deal with denoising)
        pre_load_weights(boolean):whether to start by loading some weights from another model
        pretrained_model:keras model object from where the weigths must be extracted
        plot_history(boolean): indicating whether to plot the train and validation loss graphs

      
      Output:
      history: containing the training of the model
      model: keras model trained for a particular task
  """
  #Here we create the training and validation generators 
  # define data generators to do data augmentation 
  train_generator, val_generator = get_train_val_generators( X_train,
                                                          Y_train,
                                                         X_val,Y_val,
                                                          rescale= None,
                                                          horizontal_flip=True,
                                                          vertical_flip=True,
                                                          rotation_range = None,
                                                          #width_shift_range=0.2,
                                                          #height_shift_range=0.2,
                                                          #shear_range=0.2,
                                                          preprocessing_function=random_90rotation,
                                                          batch_size=batch_size_value,
                                                          show_examples=False )
  #Here we establish the architecture based in the input model_name
  num_filters=16
  dropout_value=0.2
  if model_name == 'UNet':
      model = UNet( filters=num_filters, dropout_value=dropout_value,
                   spatial_dropout=False, average_pooling=False, activation='elu',num_outputs=output_channels,pre_load_weights=pre_load_weights,pretrained_model=pretrained_model,train_encoder=train_encoder,bottleneck_freezing=bottleneck_freezing)
  elif model_name == 'ResUNet':
      model = ResUNet( filters=num_filters, batchnorm=False, spatial_dropout=True,
                      average_pooling=False, activation='elu', separable=False,
                      dropout_value=dropout_value,num_outputs=output_channels )
  elif model_name=='MobileNetEncoder':
      model =MobileNetEncoder(
           train_encoder=train_encoder,
           random_encoder_weights=random_encoder_weights,
           output_channels=output_channels,
          max_pooling=max_pooling,pre_load_weights=pre_load_weights,pretrained_model=pretrained_model,Denoising=Denoising)
  model.summary()

  if optimizer_name == 'SGD':
      optim =  tf.keras.optimizers.SGD(
              lr=lr, momentum=0.99, decay=0.0, nesterov=False)
  elif optimizer_name == 'Adam':
      optim = tf.keras.optimizers.Adam( learning_rate=lr )

  if loss_acronym == 'BCE':
      loss_funct = 'binary_crossentropy'
  elif loss_acronym == 'Dice':
      loss_funct = dice_loss
  elif loss_acronym == 'W_BCE_Dice':
      loss_funct = weighted_bce_dice_loss(w_bce=0.8, w_dice=0.2)
  elif loss_acronym== 'CCE':
      loss_funct= tf.keras.losses.CategoricalCrossentropy()
  elif loss_acronym=='mse':
      loss_funct='mse'
  elif loss_acronym=='mae':
      loss_funct='mean_absolute_error'
  elif loss_acronym=='SEG':
      loss_funct=loss_seg(relative_weights=seg_weights)

  if preTrain:
    eval_metric = 'mean_absolute_error'
  else:
    if loss_acronym == 'BCE':
      eval_metric = jaccard_index_final
    else:
       eval_metric = jaccard_index
      

  # compile the model with the specific optimizer, loss function and metric
  model.compile(optimizer=optim, loss=loss_funct, metrics=[eval_metric])

    # callback for early stop
  earlystopper = EarlyStopping(patience=numEpochs, verbose=1, restore_best_weights=True)

  if schedule == 'oneCycle':
      # callback for one-cycle schedule
      steps = np.ceil(len(X_train) / batch_size_value) * numEpochs
      lr_schedule = OneCycleScheduler(lr, steps)
  elif schedule == 'reduce':
      # callback to reduce the learning rate in the plateau
     lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                               patience=patience, min_lr=min_lr)
  else:
      lr_schedule = None
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=('current_model'),verbose=1,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=save_best_only)
  callbacks = [earlystopper,model_checkpoint_callback] if lr_schedule is None else [earlystopper, lr_schedule,model_checkpoint_callback]

  # train!
  history = model.fit(train_generator, validation_data=val_generator,
                      validation_steps=np.ceil(len(X_val[:,0,0,0])/batch_size_value),
                      steps_per_epoch=np.ceil(len(X_train[:,0,0,0])/batch_size_value),
                      epochs=numEpochs, callbacks=callbacks)
  print('Restoring model with best weights')
  model.load_weights(filepath=('current_model'))
  print('Done!')
  import matplotlib.pyplot as plt
  if plot_history:
    plt.figure(figsize=(14,5))

    if callable( eval_metric ):
     metric_name = eval_metric.__name__
    else:
      metric_name = eval_metric

    # summarize history for loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

     # summarize history for metrics
    plt.subplot(1, 2, 2)
    plt.plot(history.history[metric_name])
    plt.plot(history.history['val_'+metric_name])
    plt.title('model ' + metric_name)
    plt.ylabel(metric_name)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

  return history,model
