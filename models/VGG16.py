from keras import Model
from keras.layers import *

class Vgg16_Unet:

    def __init__(self, inputLayer ):

        ## Constants #
        self.WEIGHTS_PATH_NO_TOP = "vgg16_weights_th_dim_ordering_th_kernels_notop.h5"

        self.input_layer = inputLayer
        self.output_layer = self.build_Vgg16Unet(inputLayer)
        self.model = Model(self.input_layer, self.output_layer)

    def return_Model(self):
            return self.model

    def build_Vgg16Unet (self, inputLayer):

        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputLayer)
        block1_output = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(block1_output)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        block2_output = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(block2_output)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        block3_output = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(block3_output)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        block4_output = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(block4_output)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        block5_output = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(block5_output)

        model = Model(inputLayer, x)
        model.load_weights(self.WEIGHTS_PATH_NO_TOP)

        x = model.get_layer('block5_pool').output
        block1_output = model.get_layer('block1_conv2').output
        block2_output = model.get_layer('block2_conv2').output
        block3_output = model.get_layer('block3_conv2').output
        block4_output = model.get_layer('block4_conv2').output
        block5_output = model.get_layer('block5_conv2').output

        ## VGG16 decoder ##

        # middle #
        x = Conv2D(512, (3,3), activation='relu', padding='same', name='block6_conv1')(x)
        x = BatchNormalization()(x)

        # upblock 5 #
        x = Conv2DTranspose(512, (3, 3), strides=(2, 2), activation="relu", padding="same", name="upblock5_Upsample")(x)
        x = concatenate([x, block5_output], name="upblock5_concate")
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='upblock5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='upblock5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='upblock5_conv3')(x)

        # upblock4 #
        x = Conv2DTranspose(512, (3, 3), strides=(2, 2), activation="relu", padding="same", name="upblock4_Upsample")(x)
        x = concatenate([x, block4_output], name="upblock4_concate")
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='upblock4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='upblock4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='upblock4_conv3')(x)

        # upblock3 #
        x = Conv2DTranspose(256, (3, 3), strides=(2, 2), activation="relu", padding="same", name="upblock3_Upsample")(x)
        x = concatenate([x, block3_output], name="upblock3_concate")
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='upblock3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='upblock3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='upblock3_conv3')(x)

        # upblock2 #
        x = Conv2DTranspose(128, (3, 3), strides=(2, 2), activation="relu", padding="same", name="upblock2_Upsample")(x)
        x = concatenate([x, block2_output], name="upblock2_concate")
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='upblock2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='upblock2_conv2')(x)

        # upblock1 #
        x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation="relu", padding="same", name="upblock1_Upsample")(x)
        x = concatenate([x, block1_output], name="upblock1_concate")
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='upblock1_conv1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='upblock1_conv2')(x)

        return Conv2D(1, (1, 1), padding="same", strides=(1, 1), activation="sigmoid")(x)


