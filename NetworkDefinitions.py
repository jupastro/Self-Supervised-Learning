# Network definitions

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, UpSampling2D, Activation
from tensorflow.keras.layers import Dropout, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D, Conv1D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Concatenate, Add, concatenate, Lambda
from tensorflow_examples.models.pix2pix import pix2pix

def MobileNetEncoder(input_size = (None,None,1),
         train_encoder=False,
         random_encoder_weights=True,
         output_channels=1,max_pooling=True,pre_load_weights=False,pretrained_model=None,Denoising=False):
  """
  Create an encoder based in MobileNet attached to a general decoder for segmentation
       Args:
            input_size (array of 3 int): dimensions of the input image.
            random_encoder_weights(bool,optional):whether to initialise the encoder's weights
               to random weights or the pretrained in the imagenet or to load previously trained ones
            Output_channels(int,optional):define the kind of segmentation(semantic) 
            and number of elements to segmentate
            max_pooling(boolean,optional):whether to apply a max_pooling or average pooling
            pre_load_weights(boolean,optional): if we want to add to our model some pretrained weights for the previous layers
            pretrained_model:model that is going to act as the starting point for our new model
       Returns:
            model (Keras model): model containing the segmentation net created.
  """
  
    #Now we load the base MobileNetV2 architecture for the decoder
  if random_encoder_weights==False:
                  input_size=(None,None,3)
  encoder_model = tf.keras.applications.MobileNetV2(input_shape=input_size, include_top=False,
                                                    weights=None if random_encoder_weights else 'imagenet',
                                                    pooling='max'if max_pooling else 'avg')

    # Use the activations of these layers as the skip connections(blocks 1-13) and bottleneck(block 16)
  layer_names = [
     'block_1_expand_relu',   
     'block_3_expand_relu',   
     'block_6_expand_relu',   
     'block_13_expand_relu', 
     'block_16_project',      
  ]
    #Now we select the previous layers
  layers = [encoder_model.get_layer(name).output for name in layer_names]
  
    # Create the feature extraction model
  down_stack = tf.keras.Model(inputs=encoder_model.input, outputs=layers)
    #Here we define the number of layers for the decoder
    # The function applies a convolution to recreate the image
  up_stack = [
    pix2pix.upsample(512, 3),  # 8x8 -> 16x16
    pix2pix.upsample(256, 3),  # 16x16 -> 32x32
    pix2pix.upsample(128, 3),  # 32x32 -> 64x64
    pix2pix.upsample(64, 3),   # 64x64 -> 128x128
  ]
# we set the whole encoder to be trainable or not
  down_stack.trainable = train_encoder
  encoder_model.trainable=train_encoder
  
  inputs = tf.keras.layers.Input(shape=input_size)
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model can be meant for denoising or classification
  if Denoising or output_channels==1:
    last_denoising = tf.keras.layers.Conv2DTranspose(
        1, 3, strides=2,
        padding='same',activation='sigmoid')  #128x128 -> 256x256
    x = last_denoising(x)
         
  
  else:
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same',activation='softmax')  #128x128 -> 256x256
    x = last(x)
    
  model= tf.keras.Model(inputs=inputs, outputs=x)#Recreates a model setting the specific layer(softmax or sigmoid act function)
  model.trainable=True
  if pre_load_weights:
    #Loading weights layer by layer except from the last layer whose structure would change 
    for i in range((len(model.layers)-1)):
        model.get_layer(index=i).set_weights(pretrained_model.get_layer(index=i).get_weights())
        print('Loaded pre-trained weights from layer',i,'of',len(model.layers))
  
  
  return model

  # Regular U-Net
def UNet(input_size = (None,None,1),
         filters=16,
         activation='elu',
         kernel_initializer = 'he_normal',
         dropout_value=0.2,
         average_pooling=True,
         spatial_dropout=False,num_outputs=1,pre_load_weights=False,pretrained_model=None,train_encoder=True,bottleneck_freezing=False):
  """
  Create a U-Net for segmentation
       Args:
            input_size (array of 3 int): dimensions of the input image.
            filters (int, optional): number of channels at the first level of U-Net
            activation (str, optional): Keras available activation type.
            kernel_initializer (str, optional): Keras available kernel 
                initializer type.
            dropout_value (real value/list/None, optional): dropout value of each
                level and the bottleneck
            average_pooling (bool, optional): use average-pooling between U-Net
                levels (otherwise use max pooling).
            spatial_dropout (bool, optional): use SpatialDroput2D, otherwise regular Dropout
            train_encoder(bool): set to true if not specified, whether to train the encoder or to freeze its weights.
       Returns:
            model (Keras model): model containing the ResUNet created.
  """
  # make a list of dropout values if needed
  if type( dropout_value ) is float:
            dropout_value = [dropout_value]*5

  inputs = Input( input_size )
  # Encoder 
  conv1 = Conv2D(filters, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(inputs)
  conv1 = SpatialDropout2D(dropout_value[0])(conv1) if spatial_dropout else Dropout(dropout_value[0]) (conv1)
  conv1 = Conv2D(filters, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(conv1)
  pool1 = AveragePooling2D(pool_size=(2, 2))(conv1) if average_pooling else MaxPooling2D(pool_size=(2, 2))(conv1)
  
  conv2 = Conv2D(filters*2, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(pool1)
  conv2 = SpatialDropout2D(dropout_value[1])(conv2) if spatial_dropout else Dropout(dropout_value[1]) (conv2)
  conv2 = Conv2D(filters*2, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(conv2)
  pool2 = AveragePooling2D(pool_size=(2, 2))(conv2) if average_pooling else MaxPooling2D(pool_size=(2, 2))(conv2)
  
  conv3 = Conv2D(filters*4, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(pool2)
  conv3 = SpatialDropout2D(dropout_value[2])(conv3) if spatial_dropout else Dropout(dropout_value[2]) (conv3)
  conv3 = Conv2D(filters*4, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(conv3)
  pool3 = AveragePooling2D(pool_size=(2, 2))(conv3) if average_pooling else MaxPooling2D(pool_size=(2, 2))(conv3)
  
  conv4 = Conv2D(filters*8, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(pool3)
  conv4 = SpatialDropout2D(dropout_value[3])(conv4) if spatial_dropout else Dropout(dropout_value[3])(conv4)
  conv4 = Conv2D(filters*8, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(conv4)
  pool4 = AveragePooling2D(pool_size=(2, 2))(conv4) if average_pooling else MaxPooling2D(pool_size=(2, 2))(conv4)

  # Bottleneck
  conv5 = Conv2D(filters*16, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(pool4)
  conv5 = SpatialDropout2D(dropout_value[4])(conv5) if spatial_dropout else Dropout(dropout_value[4])(conv5)
  conv5 = Conv2D(filters*16, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(conv5)
  
  # Decoder
  up6 = Conv2DTranspose(filters*8, (2, 2), strides=(2, 2), padding='same') (conv5)
  merge6 = concatenate([conv4,up6], axis = 3)
  conv6 = Conv2D(filters*8, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(merge6)
  conv6 = SpatialDropout2D(dropout_value[3])(conv6) if spatial_dropout else Dropout(dropout_value[3])(conv6)
  conv6 = Conv2D(filters*8, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(conv6)

  up7 = Conv2DTranspose(filters*4, (2, 2), strides=(2, 2), padding='same') (conv6)
  merge7 = concatenate([conv3,up7], axis = 3)
  conv7 = Conv2D(filters*4, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(merge7)
  conv7 = SpatialDropout2D(dropout_value[2])(conv7) if spatial_dropout else Dropout(dropout_value[2])(conv7)
  conv7 = Conv2D(filters*4, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(conv7)

  up8 = Conv2DTranspose(filters*2, (2, 2), strides=(2, 2), padding='same') (conv7)
  merge8 = concatenate([conv2,up8], axis = 3)
  conv8 = Conv2D(filters*2, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(merge8)
  conv8 = SpatialDropout2D(dropout_value[1])(conv8) if spatial_dropout else Dropout(dropout_value[1])(conv8)
  conv8 = Conv2D(filters*2, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(conv8)

  up9 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same') (conv8)
  merge9 = concatenate([conv1,up9], axis = 3)
  conv9 = Conv2D(filters, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(merge9)
  conv9 = SpatialDropout2D(dropout_value[0])(conv9) if spatial_dropout else Dropout(dropout_value[0])(conv9)
  conv9 = Conv2D(filters, (3,3), activation = activation, padding = 'same', kernel_initializer = kernel_initializer)(conv9)
  if num_outputs==1:
         outputs = Conv2D( num_outputs, (1, 1), activation='sigmoid') (conv9)
  else:
         outputs = Conv2D( num_outputs, (1, 1), activation='softmax') (conv9)
  
                  
  model = Model(inputs=[inputs], outputs=[outputs])
  if pre_load_weights:
    #Loading weights layer by layer except from the last layer whose structure would change 
  
      for i in range((len(model.layers)-1)):
        model.get_layer(index=i).set_weights(pretrained_model.get_layer(index=i).get_weights())
        print('Loaded pre-trained weights from layer',i,'of',len(model.layers))
  if train_encoder==False:
        for i in range(0,16):  
         model.get_layer(index=i).trainable=False
        print('The encoder has been succesfully frozen')
        if bottleneck_freezing:
         model.get_layer(index=16).trainable=False
         print('The bottleneck has been succesfully frozen')
  #for layer in model.layers:
   #      print(layer, layer.trainable)

  return model


# == Residual U-Net ==

def residual_block(x, dim, filter_size, activation='elu', 
                   kernel_initializer='he_normal', dropout_value=0.2, bn=False,
                   separable_conv=False, firstBlock=False, spatial_dropout=False):

    # Create shorcut
    shortcut = Conv2D(dim, activation=None, kernel_size=(1, 1), 
                      strides=1)(x)
    
    # Main path
    if firstBlock == False:
        x = BatchNormalization()(x) if bn else x
        x = Activation( activation )(x)
    if separable_conv == False or firstBlock:
        x = Conv2D(dim, filter_size, strides=1, activation=None,
                kernel_initializer=kernel_initializer, padding='same') (x)
    else:
        x = SeparableConv2D(dim, filter_size, strides=1, 
                            activation=None, kernel_initializer=kernel_initializer,
                            padding='same') (x)
    if dropout_value:
        x = SpatialDropout2D( dropout_value ) (x) if spatial_dropout else Dropout( dropout_value ) (x)
        print( str( dropout_value ) )
    x = BatchNormalization()(x) if bn else x
    x = Activation( activation )(x)
      
    if separable_conv == False:
        x = Conv2D(dim, filter_size, activation=None,
                kernel_initializer=kernel_initializer, padding='same') (x)
    else:
        x = SeparableConv2D(dim, filter_size, activation=None,
                kernel_initializer=kernel_initializer, padding='same') (x)

    # Add shortcut value to main path
    x = Add()([shortcut, x])
    print( 'residual block, dim: ' + str(dim) + ' , output shape: '+ str(x.shape) )
    return x

def level_block(x, depth, dim, fs, ac, k, d, bn, sc, fb, ap, spatial_dropout):
    do = d[depth] if d is not None else None
    if depth > 0:
        r = residual_block(x, dim, fs, ac, k, do, bn, sc, fb, spatial_dropout)
        x = AveragePooling2D((2, 2)) (r) if ap else MaxPooling2D((2, 2)) (r)
        x = level_block(x, depth-1, (dim*2), fs, ac, k, d, bn, sc, False, ap, spatial_dropout) 
        x = Conv2DTranspose(dim, (2, 2), strides=(2, 2), padding='same') (x)
        x = Concatenate()([r, x])
        x = residual_block(x, dim, fs, ac, k, do, bn, sc, False, spatial_dropout)
    else:
        x = residual_block(x, dim, fs, ac, k, do, bn, sc, False, spatial_dropout)
    return x


def ResUNet( input_size=(None, None, 1), activation='elu', kernel_initializer='he_normal',
            dropout_value=0.2, batchnorm=False, average_pooling=False, separable=False,
            filters=16, depth=4, spatial_dropout=False, long_shortcut=True,num_outputs=1):

    """Create a Residual U-Net for segmentation
       Args:
            input_size (array of 3 int): dimensions of the input image.
            activation (str, optional): Keras available activation type.
            kernel_initializer (str, optional): Keras available kernel 
            initializer type.
            dropout_value (real value/list/None, optional): dropout value of each
            level and the bottleneck
            batchnorm (bool, optional): use batch normalization
            average_pooling (bool, optional): use average-pooling between U-Net levels 
            (otherwise use max pooling).
            separable (bool, optional): use SeparableConv2D instead of Conv2D
            filters (int, optional): number of channels at the first level of U-Net
            depth (int, optional): number of U-Net levels
            spatial_dropout (bool, optional): use SpatialDroput2D, otherwise regular Dropout
            long_shortcut (bool, optional): add long shorcut from input to output.
       Returns:
            model (Keras model): model containing the ResUNet created.
    """

    inputs = Input( input_size )
    if dropout_value is not None:
        if type( dropout_value ) is float:
            dropout_value = [dropout_value]*(depth+1)
        else:
            dropout_value.reverse() # reverse list to go from top to down

    x = level_block(inputs, depth, filters, 3, activation, kernel_initializer,
                    dropout_value, batchnorm, separable, True, average_pooling,
                    spatial_dropout)

    if long_shortcut:
        x = Add()([inputs,x]) # long shortcut
    if num_outputs==1:
         outputs = Conv2D( num_outputs, (1, 1), activation='sigmoid' ) (x)
    else:
         outputs = Conv2D( num_outputs, (1, 1), activation='softmax' ) (x)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model
