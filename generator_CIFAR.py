import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

sys.path.append(os.path.abspath(os.path.join('..', '')))

from TraineeModels.setup_mnist import *
from TraineeModels.setup_cifar import *
from generator_v2 import *
from discriminator_v2 import *


class MNISTGenerator:
    def __init__(self):
        self.optimizer = tf.keras.optimizers.Adam(1e-6)
        self.model = self.make_generator_model()

    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        return model

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)


class CIFARGenerator:
    def __init__(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1=0.5)
        self.model = self.make_generator_model()


    def make_generator_model(self):
        model = tf.keras.Sequential()
        # model.add(layers.Dense(4 * 2 * 256))
        # model.add(layers.LeakyReLU(alpha=0.2))
        #
        # model.add(layers.Reshape((4, 4, 256)))
        #assert model.output_shape == (None, 4, 4, 256)  # Note: None is the batch size

        #model.add(layers.Conv2D(64, (3, 3), activation="relu", input_shape=[32, 32, 3]))
        #model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        #model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        # model.add(layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same'))
        # #assert model.output_shape == (None, 8, 8, 128)
        # model.add(layers.LeakyReLU(alpha=0.2))
        #
        # model.add(layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same'))
        # #assert model.output_shape == (None, 16, 16, 128)
        # model.add(layers.LeakyReLU(alpha=0.2))
        #
        # model.add(layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same'))
        # model.add(layers.LeakyReLU(alpha=0.2))
        # #assert model.output_shape == (None, 32, 32, 128)
        #
        # model.add(layers.Conv2D(3, (3, 3), activation='tanh', padding='same'))
        initializer = tf.keras.initializers.GlorotUniform()

        model.add(layers.Conv2D(48, (3, 3), padding='same', kernel_initializer=initializer, input_shape=[32, 32, 3]))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Conv2D(48, (3, 3), padding='same', kernel_initializer=initializer))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Conv2D(48, (3, 3), padding='same', kernel_initializer=initializer))
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2D(48, (3, 3), padding='same', kernel_initializer=initializer))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Conv2D(48, (3, 3), padding='same', kernel_initializer=initializer))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Conv2D(48, (3, 3), padding='same', kernel_initializer=initializer))
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2D(48, (1, 1), padding='same', kernel_initializer=initializer))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Conv2D(3, (1, 1), activation='tanh'))

        #model.compile(loss='categorical_crossentropy', optimizer=self.optimizer)

        return model
