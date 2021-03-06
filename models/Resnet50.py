
""" Heavily based on Siddhartha's ResNet50 code
https://www.kaggle.com/meaninglesslives/using-resnet50-pretrained-model-in-keras 
Merged into class for increased deployability
"""

from keras import Model
from keras.layers import *
from keras import layers

from keras import Model
from keras.layers import *
from keras import layers

from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file


class Resnet50_Unet():

    def __init__(self, inputLayer, include_top= False, weights= None):
        
        self.WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
        self.WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



    def conv_block_simple(self, prevlayer, filters, prefix, strides=(1, 1)):
        conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
        conv = BatchNormalization(name=prefix + "_bn")(conv)
        conv = Activation('relu', name=prefix + "_activation")(conv)
        return conv

    def conv_block_simple_no_bn(self, prevlayer, filters, prefix, strides=(1, 1)):
        conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
        conv = Activation('relu', name=prefix + "_activation")(conv)
        return conv
    K.clear_session()

    def get_unet_resnet(self, input_shape, include_top = False,weights=None ):
        
        resnet_base = self.ResNet50(input_shape=input_shape, include_top= include_top, weights=weights)

        for l in resnet_base.layers:
            l.trainable = True
        conv1 = resnet_base.get_layer("activation_1").output
        conv2 = resnet_base.get_layer("activation_10").output
        conv3 = resnet_base.get_layer("activation_22").output
        conv4 = resnet_base.get_layer("activation_40").output
        conv5 = resnet_base.get_layer("activation_49").output

        up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
        conv6 = self.conv_block_simple(up6, 256, "conv6_1")
        conv6 = self.conv_block_simple(conv6, 256, "conv6_2")

        up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
        conv7 = self.conv_block_simple(up7, 192, "conv7_1")
        conv7 = self.conv_block_simple(conv7, 192, "conv7_2")

        up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
        conv8 = self.conv_block_simple(up8, 128, "conv8_1")
        conv8 = self.conv_block_simple(conv8, 128, "conv8_2")

        up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
        conv9 = self.conv_block_simple(up9, 64, "conv9_1")
        conv9 = self.conv_block_simple(conv9, 64, "conv9_2")

        up10 = UpSampling2D()(conv9)
        conv10 = self.conv_block_simple(up10, 32, "conv10_1")
        conv10 = self.conv_block_simple(conv10, 32, "conv10_2")
        conv10 = SpatialDropout2D(0.2)(conv10)
        x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
        model = Model(resnet_base.input, x)
    
        return model

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        
        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,
                padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
        
        return x


    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

        filters1, filters2, filters3 = filters
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
    
        x = Conv2D(filters1, (1, 1), strides=strides,
                   name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(filters2, kernel_size, padding='same',
                   name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    
        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    
        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x
    
    def ResNet50(self, include_top=False, weights=None,
                 input_tensor=None, input_shape=None,
                 pooling=None,
                 classes=1000):
        if weights not in {'imagenet', None}:
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `imagenet` '
                             '(pre-training on ImageNet).')
    
        if weights == 'imagenet' and include_top and classes != 1000:
            raise ValueError('If using `weights` as imagenet with `include_top`'
                             ' as true, `classes` should be 1000')
    
        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
    
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    
        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    
        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    
        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    
        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    
    
        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input
        # Create model.
        model = Model(inputs, x, name='resnet50')
    
        # load weights
        if weights == 'imagenet':
            if include_top:
                weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                        self.WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
            else:
                weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        self.WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='a268eb855778b3df3c7506639542a6af')
            model.load_weights(weights_path,by_name=True)
        return model


