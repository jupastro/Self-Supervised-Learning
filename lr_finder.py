
#Implementation based in the tutorial of https://www.pyimagesearch.com/2019/08/05/keras-learning-rate-finder/
# import the necessary packages
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tempfile
from SelfSupervisedLearning.Data_aug import get_train_val_generators,random_90rotation

class lr_finder:

  def __init__(self, model, stopFactor=4, beta=0.98):
    self.model = model
    self.stopFactor = stopFactor
    self.beta = beta
    self.learning_rates = []
    self.losses = []  
    self.lrMult = 1
    self.avgLoss = 0
    self.bestLoss = 1e9
    self.n_batch = 0
    self.weightsFile = None
   

  def reset(self):
    #it is a method that allows the class to reset all the variables once finished
    self.learning_rates = []
    self.losses = []
		# initialize our learning rate multiplier, average loss, best
		# loss found thus far, current batch number, and weights file
    self.lrMult = 1
    self.avgLoss = 0
    self.bestLoss = 1e9
    self.n_batch = 0
    self.weightsFile = None

  def on_batch_end(self, batch, logs):
		# grab the current learning rate and add log it to the list of
		# learning rates that we've tried
    current_lr=K.get_value(self.model.optimizer.lr)
    self.learning_rates.append(current_lr)
		# grab the loss at the end of this batch, increment the total
		# number of batches processed, compute the average average
		# loss, smooth it, and update the losses list with the
		# smoothed value
    l=logs['loss']
    self.n_batch+=1
    self.avg_loss=(self.beta * self.avgLoss) + ((1 - self.beta) * l)#represents a weighted sum where beta is the importance of the last loss value
    smooth=self.avg_loss/(1-self.beta**self.n_batch)
    self.losses.append(smooth)
    #self.losses.append(l)
    

	
		# compute the maximum loss stopping factor value
    stop_loss=self.bestLoss*self.stopFactor
		# check to see whether the loss has grown too large
    if smooth>stop_loss and self.n_batch>1:
      self.model.stop_training=True
			# stop returning and return from the method
			
      return
		# check to see if the best loss should be updated
    if self.n_batch==1 and smooth<self.bestLoss:
      self.bestLoss = smooth
		# increase the learning rate
    
    current_lr *= self.lrMult
    K.set_value(self.model.optimizer.lr, current_lr)
  
  def find(self, Xtrain,Ytrain,Xval,Yval, startLR, endLR, epochs=None,
		stepsPerEpoch=None, batchSize=32,
		verbose=1):
		# reset our class-specific variables
    self.reset()

		# grab the number of samples in the training data and
		# then derive the number of steps per epoch
    numSamples = len(Ytrain)
    if stepsPerEpoch is None:
      stepsPerEpoch = np.ceil(numSamples / float(batchSize))
		# if no number of training epochs are supplied, compute the
		# training epochs based on a default sample size
    if epochs is None:
      epochs = int(np.ceil(sampleSize / float(stepsPerEpoch)))
   # compute the total number of batch updates that will take
		# place while we are attempting to find a good starting
		# learning rate
    numBatchUpdates = epochs * stepsPerEpoch
		# derive the learning rate multiplier based on the ending
		# learning rate, starting learning rate, and total number of
		# batch updates
    self.lrMult = (endLR / startLR) ** (1.0 / numBatchUpdates)
		# create a temporary file path for the model weights and
		# then save the weights (so we can reset the weights when we
		# are done)
    self.weightsFile = tempfile.mkstemp()[1]
    self.model.save_weights(self.weightsFile)
		# grab the *original* learning rate (so we can reset it
		# later), and then set the *starting* learning rate
    origLR = K.get_value(self.model.optimizer.lr)
    K.set_value(self.model.optimizer.lr, startLR)
  	# construct a callback that will be called at the end of each
		# batch, enabling us to increase our learning rate as training
		# progresses
    callback = LambdaCallback(on_batch_end=lambda batch, logs:
                              self.on_batch_end(batch, logs))
		# check to see if we are using a data iterator
    train_generator, val_generator = get_train_val_generators( Xtrain,
                                                          Ytrain,
                                                         Xval,Yval,
                                                          rescale= None,
                                                          horizontal_flip=True,
                                                          vertical_flip=True,
                                                          rotation_range = None,
                                                          #width_shift_range=0.2,
                                                          #height_shift_range=0.2,
                                                          #shear_range=0.2,
                                                          preprocessing_function=random_90rotation,
                                                          batch_size=batchSize,
                                                          show_examples=False )
    
    self.model.fit(train_generator, validation_data=val_generator,
                      validation_steps=np.ceil(len(Xval[:,0,0,0])/batchSize),
                      steps_per_epoch=stepsPerEpoch,
                      epochs=epochs, callbacks=callback)
		# restore the original model weights and learning rate
    self.model.load_weights(self.weightsFile)
    K.set_value(self.model.optimizer.lr, origLR)


  def plot_loss(self, skipBegin=15, skipEnd=1, title=""):
    # grab the learning rate and losses values to plot
    lrs = self.learning_rates[skipBegin:-skipEnd]
    # The first steps due to random initialisation of the weights there is almost sure a huge decrease of the loss 
    #Not being representative of a good lr
    losses = self.losses[skipBegin:-skipEnd]
    d=[losses[i]-losses[i-1] for i in range(1,len(losses))]
 # plot the learning rate vs. loss
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1),plt.plot(smoothing(lrs,10), smoothing(losses,10))
    plt.xscale("log")
    plt.xlabel("Learning Rate (Log Scale)")
    plt.ylabel("Loss")
    plt.subplot(1,2,2),plt.plot(smoothing(lrs[1:],10), smoothing(d,10))
#plot the gradient of the loss vs lr
    plt.xscale("log")
    plt.xlabel("Learning Rate (Log Scale)")
    plt.ylabel("Loss derivative")
    min_idx=d.index(min(d))
    self.optMinlr=lrs[min_idx]
#to look within the biggest values that are smaller than 0 
    possible_list=d[min_idx:]
    possible_max=[]
    for i in possible_list:
      if i<0:
        possible_max.append(i)
      elif i>0:
        break
			       
    max_dloss=max(possible_max)
    self.optMaxlr=lrs[d.index(max_dloss)]
    print('The suggested min lr with the maximal negative loss gradient is: '+ str(self.optMinlr))
    print('The suggested max lr with 0 loss gradient is: '+ str(self.optMaxlr))
    # if the title is not empty, add it to the plot
    if title != "":
      plt.suptitle(title)
    return lrs,losses,d



def smoothing(x,y):
	return np.convolve(x, np.ones(y), 'same')/y
