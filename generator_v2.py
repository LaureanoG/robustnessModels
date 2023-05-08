import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class MNISTGenerator:
    def __init__(self):
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
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

        return model

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)


class CIFARGenerator:
    def __init__(self):
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0002, beta_1=0.5)
        self.model = self.make_generator_model()


    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(4 * 4 * 256, input_shape=(100,)))
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Reshape((4, 4, 256)))
        assert model.output_shape == (None, 4, 4, 256)  # Note: None is the batch size

        model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 8, 8, 128)
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        assert model.output_shape == (None, 16, 16, 128)
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        assert model.output_shape == (None, 32, 32, 128)

        model.add(layers.Conv2D(3, (3, 3), activation='tanh', padding='same'))
        model.compile(loss='binary_crossentropy', optimizer=self.generator_optimizer)

        return model

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)
