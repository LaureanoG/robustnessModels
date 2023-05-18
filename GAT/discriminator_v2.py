import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam
from TraineeModels.setup_mnist import *
from TraineeModels.setup_cifar import *
from gat_train_v3 import *


class MNISTDiscriminator:
    def __init__(self):
        self.optimizer = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model = self.make_discriminator_model()

    def make_discriminator_model(self):
        model = tf.keras.models.Sequential()

        model.add(layers.Conv2D(32, (3, 3),
                         input_shape=(28, 28, 1)))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(32, (3, 3)))
        model.add(layers.LeakyReLU())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(64, (3, 3)))
        model.add(layers.LeakyReLU())
        model.add(layers.Conv2D(64, (3, 3)))
        model.add(layers.LeakyReLU())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(200))
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(200))
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(10, activation="softmax"))

        return model


class CIFARDiscriminator:
    def __init__(self):
        self.optimizer = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self.model = self.make_discriminator_model()

    def make_discriminator_model(self):
        model = tf.keras.models.Sequential()
        model.add(layers.Conv2D(64, (3, 3),  input_shape=[32, 32, 3]))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Conv2D(64, (3, 3)))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        #model.add(layers.AveragePooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(128, (3, 3)))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Conv2D(128, (3, 3)))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        #model.add(layers.AveragePooling2D(pool_size=(2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(256))
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(256))
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(10, activation="softmax"))

        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        return model

