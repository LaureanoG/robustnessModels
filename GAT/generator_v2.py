import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

sys.path.append(os.path.abspath(os.path.join('..', '')))

from TraineeModels.setup_mnist import *
from TraineeModels.setup_cifar import *
from generator_v2 import *
from discriminator_v2 import *
from gat_train_v3 import *


class MNISTGenerator:
    def __init__(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1=0.5)
        self.model = self.make_generator_model()

    def make_generator_model(self):
        initializer = tf.keras.initializers.GlorotUniform()

        model = tf.keras.Sequential()
        model.add(layers.Conv2D(48, (3, 3), padding='same', kernel_initializer=initializer, input_shape=[28, 28, 1]))
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
        model.add(layers.Conv2D(1, (1, 1), activation='tanh'))

        model.compile(loss=GAN_loss, optimizer=self.optimizer)

        return model


class CIFARGenerator:
    def __init__(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1=0.5)
        self.model = self.make_generator_model()


    def make_generator_model(self):
        initializer = tf.keras.initializers.GlorotUniform()

        model = tf.keras.Sequential()
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

        model.compile(loss=GAN_loss, optimizer=self.optimizer)

        return model
