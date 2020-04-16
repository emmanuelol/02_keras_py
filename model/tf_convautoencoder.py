# -*- coding: utf-8 -*-
"""
オートエンコーダ
https://www.pyimagesearch.com/2020/02/17/autoencoders-with-keras-tensorflow-and-deep-learning/

Usage:
    EPOCHS = 25
    BS = 32
    width, height, depth = 100, 100, 3
    filters = (32, 64) # この2層のfilterならinputとoutput同じサイズになる、層増やすとサイズずれてエラーになる
    latentDim = 256#16
    (encoder, decoder, autoencoder) = tf_ConvAutoencoder.build(width, height, depth, filters=filters, latentDim=latentDim)
    opt = Adam(lr=1e-3)
    autoencoder.compile(loss="mse", optimizer=opt)
    H = autoencoder.fit(trainX, trainX
                        , validation_data=(testX, testX)
                        , epochs=EPOCHS
                        , batch_size=BS)
    decoded = autoencoder.predict(testX)
"""
import numpy as np
from tensorflow.keras import backend as K
# import the necessary packages
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    Input,
    LeakyReLU,
    Reshape
)
from tensorflow.keras.models import Model


class ConvAutoencoder:
    @staticmethod
    def build(width, height, depth, filters=(32, 64), latentDim=16):
        # initialize the input shape to be "channels last" along with
        # the channels dimension itself
        # channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1

        # define the input to the encoder
        inputs = Input(shape=inputShape)
        x = inputs

        # loop over the number of filters
        for f in filters:
            # apply a CONV => RELU => BN operation
            x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)

        # flatten the network and then construct our latent vector
        volumeSize = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latentDim)(x)

        # build the encoder model
        encoder = Model(inputs, latent, name="encoder")

        # start building the decoder model which will accept the
        # output of the encoder as its inputs
        latentInputs = Input(shape=(latentDim,))
        x = Dense(np.prod(volumeSize[1:]))(latentInputs)
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

        # loop over our number of filters again, but this time in
        # reverse order
        for f in filters[::-1]:
            # apply a CONV_TRANSPOSE => RELU => BN operation
            x = Conv2DTranspose(f, (3, 3), strides=2,
                padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)

        # apply a single CONV_TRANSPOSE layer used to recover the
        # original depth of the image
        x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
        outputs = Activation("sigmoid")(x)

        # build the decoder model
        decoder = Model(latentInputs, outputs, name="decoder")

        # our autoencoder is the encoder + decoder
        autoencoder = Model(inputs, decoder(encoder(inputs)),
            name="autoencoder")

        # return a 3-tuple of the encoder, decoder, and autoencoder
        return (encoder, decoder, autoencoder)
