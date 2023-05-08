import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class MNISTDiscriminator:
    def __init__(self):
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.model = self.make_discriminator_model()

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                input_shape=[28, 28, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss


class CIFARDiscriminator:
    def __init__(self):
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1 = 0.5)
        self.model = self.make_discriminator_model()

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        # model.add(layers.Conv2D(64, (3, 3), activation="relu", input_shape=[32, 32, 3]))
        # model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        #
        # model.add(layers.Conv2D(128, (3, 3), activation="relu"))
        # model.add(layers.Conv2D(128, (3, 3), activation="relu"))
        # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        #
        # model.add(layers.Flatten())
        # model.add(layers.Dense(256, activation="relu"))
        # model.add(layers.Dense(256, activation="relu"))
        # model.add(layers.Dense(10, activation="softmax"))

        model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=[32, 32, 3]))
        model.add(layers.LeakyReLU(alpha=0.2))
        # downsample
        model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        # downsample
        model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        # downsample
        model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        # classifier
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=self.discriminator_optimizer, metrics=['accuracy'])

        return model

    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
