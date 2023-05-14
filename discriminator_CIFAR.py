import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam
from TraineeModels.setup_mnist import *
from TraineeModels.setup_cifar import *


class MNISTDiscriminator:
    def __init__(self):
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
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
        self.optimizer = Adam(learning_rate=1e-3, beta_1 = 0.5)
        self.model = self.make_discriminator_model()

    def make_discriminator_model(self):
        model = tf.keras.models.Sequential()
        # model.add(layers.Conv2D(64, (3, 3),  input_shape=[32, 32, 3]))
        # model.add(layers.LeakyReLU())
        # model.add(layers.Conv2D(64, (3, 3)))
        # model.add(layers.LeakyReLU())
        # #model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        # model.add(layers.AveragePooling2D(pool_size=(2, 2)))
        #
        # model.add(layers.Conv2D(128, (3, 3)))
        # model.add(layers.LeakyReLU())
        # model.add(layers.Conv2D(128, (3, 3)))
        # model.add(layers.LeakyReLU())
        # #model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        # model.add(layers.AveragePooling2D(pool_size=(2, 2)))
        #
        # model.add(layers.Flatten())
        # model.add(layers.Dense(256))
        # model.add(layers.LeakyReLU())
        # model.add(layers.Dense(256))
        # model.add(layers.LeakyReLU())
        # model.add(layers.Dense(10, activation="softmax"))
        initializer = tf.keras.initializers.GlorotUniform()

        model.add(layers.Conv2D(48, (3, 3), kernel_initializer=initializer, input_shape=[32, 32, 3]))
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2D(48, (3, 3), strides=(2, 2), kernel_initializer=initializer))
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2D(96, (3, 3), kernel_initializer=initializer))
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2D(96, (3, 3), strides=(2, 2), kernel_initializer=initializer))
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2D(96, (3, 3), kernel_initializer=initializer))
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2D(96, (1, 1), kernel_initializer=initializer))
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2D(10, (1, 1), padding='same', activation='softmax'))
        model.add(tf.keras.layers.GlobalAveragePooling2D())

        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        return model

    # def discriminator_loss(self, real_output, fake_output, image_labels):
    #     cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    #     real_loss = cross_entropy(image_labels, real_output)
    #     fake_loss = cross_entropy(image_labels, fake_output)
    #     total_loss = 0.5 * real_loss + 0.5 * fake_loss
    #     return total_loss

    def discriminator_loss(self, correct, predicted):

        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                logits=predicted / 1)


class Custom_disc_acc(tf.keras.metrics.Metric):
    def __init__(self, name="acc", **kwargs):
        super(Custom_disc_acc, self).__init__(name=name, **kwargs)
        self.acc = 0.0

    def update_state(self, y_true, y_pred, sample_weight=None):
        correct = 0
        for i in len(y_pred.shape[0]):
            if y_pred[i] == y_true[i]:
                correct+=1
        self.acc = correct/y_pred.shape[0]


    def result(self):
        return self.acc

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.acc(0.0)


