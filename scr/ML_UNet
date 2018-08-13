from keras import Model
from keras.layers import *

class UNetML:

    def __init__(self, input_layer, filter_Num):

        self.input_layer = input_layer
        self.output_layer = self.buildUNetML(input_layer, filter_Num)

    def buildUNetML(self, input_layer, filterNum):

        conv1 = ML_ConvUnit(input_layer, filterNum).ConvUnit()
        maxPooling1 = MaxPool2D((2, 2))(conv1)

        conv2 = ML_ConvUnit(maxPooling1, filterNum).ConvUnit()
        conv2 = ML_ConvUnit(conv2, filterNum).ConvUnit()
        maxPooling2 = MaxPool2D((2, 2))(conv2)

        conv3 = ML_ConvUnit(maxPooling2, filterNum).ConvUnit()
        conv3 = ML_ConvUnit(conv3, filterNum).ConvUnit()
        maxPooling3 = MaxPool2D((2, 2))(conv3)

        conv4 = ML_ConvUnit(maxPooling3, filterNum).ConvUnit()
        conv4 = ML_ConvUnit(conv4, filterNum).ConvUnit()
        maxPooling4 = MaxPool2D((2, 2))(conv4)

        ## middle
        convm = ML_ConvUnit(maxPooling4, filterNum).ConvUnit()
        convm = ML_ConvUnit(conv3, filterNum).ConvUnit()

        deconv4 = Conv2DTranspose(filterNum * 8, (2, 2), strides=(2, 2), padding="same")(convm)
        uconv4 = concatenate([deconv4, conv4])
        uconv4 = ML_ConvUnit(uconv4, filterNum).ConvUnit()
        uconv4 = ML_ConvUnit(uconv4, filterNum).ConvUnit()

        deconv3 = Conv2DTranspose(filterNum * 8, (2, 2), strides=(2, 2), padding="same")(uconv4)
        uconv3 = concatenate([deconv3, conv3])
        uconv3 = ML_ConvUnit(uconv3, filterNum).ConvUnit()
        uconv3 = ML_ConvUnit(uconv3, filterNum).ConvUnit()

        deconv2 = Conv2DTranspose(filterNum * 8, (2, 2), strides=(2, 2), padding="same")(uconv3)
        uconv2 = concatenate([deconv2, conv2])
        uconv2 = ML_ConvUnit(uconv2, filterNum).ConvUnit()
        uconv2 = ML_ConvUnit(uconv2, filterNum).ConvUnit()

        deconv1 = Conv2DTranspose(filterNum * 8, (2, 2), strides=(2, 2), padding="same")(uconv2)
        uconv1 = concatenate([deconv1, conv1])
        uconv1 = ML_ConvUnit(uconv1, filterNum).ConvUnit()
        uconv1 = ML_ConvUnit(uconv1, filterNum).ConvUnit()

        output_layer = Conv2D(1, (1, 1), padding="same", activation="elu")(uconv1)

        return output_layer

    def returnModel(self):

        return Model(self.input_layer, self.output_layer)


class ML_ConvUnit:
    def __init__(self, prevLayer, filterNum ):
        self.filter_num = filterNum
        self.prevLayer = prevLayer

    def ConvUnit(self):
        conv = Conv2D(self.filter_num, (3, 3), activation="elu", padding="same")(self.prevLayer)
        batchNorm = BatchNormalization()(conv)

        return batchNorm
