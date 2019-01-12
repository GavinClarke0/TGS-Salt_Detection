from keras import Model
from keras.layers import *

class UNetBasic:

    def __init__(self, input_layer, filter_Num):

        self.input_layer = input_layer
        self.output_layer = self.buildUNet(input_layer, filter_Num)

    def buildUNet(self, inputLayer, filterNum):

        conv1 = Conv2D(filterNum, (3, 3), activation="relu", padding="same")(inputLayer)
        conv1 = Conv2D(filterNum, (3, 3), activation="relu", padding="same")(conv1)
        maxPooling1 = MaxPool2D((2,2))(conv1)

        # 1st 1/2
        conv2 = Conv2D(filterNum*2, (3, 3), activation="relu", padding="same")(maxPooling1)
        conv2 = Conv2D(filterNum*2, (3, 3), activation="relu", padding="same")(conv2)
        maxPooling2 =  MaxPool2D((2,2))(conv2)

        # 2rd 1/2
        conv3 = Conv2D(filterNum*4, (3, 3), activation="relu", padding="same")(maxPooling2)
        conv3 = Conv2D(filterNum*4, (3, 3), activation="relu", padding="same")(conv3)
        maxPooling3 = MaxPool2D((2,2))(conv3)

        # 3rd 1/2
        conv4 = Conv2D(filterNum*8, (3,3,), activation="relu", padding="same")(maxPooling3)
        conv4 = Conv2D(filterNum*8,(3,3), activation="relu", padding="same")(conv4)
        maxPooling4 = MaxPool2D((2,2))(conv4)

        # middle
        convm = Conv2D(filterNum * 16, (3, 3), activation="relu", padding="same")(maxPooling4)
        convm = Conv2D(filterNum * 16, (3, 3), activation="relu", padding="same")(convm)

        # deconv 1
        deconv4 = Conv2DTranspose(filterNum*8, (2, 2), strides=(2, 2), padding="same")(convm)
        uconv4 = concatenate([deconv4, conv4])
        uconv4 = Conv2D(filterNum * 8, (3,3),activation="relu", padding="same")(uconv4)
        uconv4 = Conv2D(filterNum * 8, (3, 3), activation="relu", padding="same")(uconv4)

        deconv3 = Conv2DTranspose(filterNum * 8, (2, 2), strides=(2, 2), padding="same")(uconv4)
        uconv3 = concatenate([deconv3, conv3])
        uconv3 = Conv2D(filterNum * 4, (3, 3), activation="relu", padding="same")(uconv3)
        uconv3 = Conv2D(filterNum * 4, (3, 3), activation="relu", padding="same")(uconv3)

        deconv2 = Conv2DTranspose(filterNum * 8, (2, 2), strides=(2, 2), padding="same")(uconv3)
        uconv2 = concatenate([deconv2, conv2])
        uconv2 = Conv2D(filterNum * 2, (3, 3), activation="relu", padding="same")(uconv2)
        uconv2 = Conv2D(filterNum * 2, (3, 3), activation="relu", padding="same")(uconv2)

        deconv1 = Conv2DTranspose(filterNum * 8, (2, 2), strides=(2, 2), padding="same")(uconv2)
        uconv1 = concatenate([deconv1, conv1])
        uconv1 = Conv2D(filterNum , (3, 3), activation="relu", padding="same")(uconv1)
        uconv1 = Conv2D(filterNum , (3, 3), activation="relu", padding="same")(uconv1)

        output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

        return output_layer

    def returnModel(self):

        return Model(self.input_layer, self.output_layer)
