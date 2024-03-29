# Laureano Griffin - U68919904
# NOTES
# GAN structure follows this tutorial: https://www.tensorflow.org/tutorials/generative/dcgan
# GAN also follows this: https://github.com/Zackory/Keras-MNIST-GAN/blob/master/mnist_dcgan.py
# GAT is from this paper: https://arxiv.org/pdf/1705.03387.pdf


import numpy as np
import sys, os, time
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data

sys.path.append(os.path.abspath(os.path.join('..', '')))

from TraineeModels.setup_mnist import *
from TraineeModels.setup_cifar import *
from generator_v2 import *
from discriminator_v2 import *

# Constants
EPOCHS = 100
BATCH_SIZE = 132
DATASET = "CIFAR10"
NOISE_DIM = 100
SEED = tf.random.normal([49, NOISE_DIM])



def generate_and_save_images(model, epoch, test_input, dataset):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  if dataset.name == "MNIST":
    for i in range(predictions.shape[0]):
        plt.subplot(7, 7, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
  else:
    for i in range(predictions.shape[0]):
        plt.subplot(7, 7, i + 1)
        #print(predictions[i, :, :, :], predictions[i, :, :, :].shape)
        # plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
        # plt.imshow(predictions[i, :, :, 1] * 127.5 + 127.5)
        # plt.imshow(predictions[i, :, :, 2] * 127.5 + 127.5)
        plt.imshow((predictions[i, :, :, :] + 1)/2)
        plt.axis('off')

  #print(predictions[0])

  plt.savefig('Images/{}_image_at_epoch_{:04d}.png'.format(DATASET, epoch))
  #plt.show()


def train_step(images, generator, discriminator):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator.model(noise, training=True)

      real_output = discriminator.model(images, training=True)
      fake_output = discriminator.model(generated_images, training=True)

      gen_loss = generator.generator_loss(fake_output)
      disc_loss = discriminator.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.model.trainable_variables)

    generator.generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.model.trainable_variables))
    discriminator.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.model.trainable_variables))
    pass


def train(dataset, generator, discriminator):
    # Initialize Checkpoints
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator.generator_optimizer,
                                     discriminator_optimizer=discriminator.discriminator_optimizer,
                                     generator=generator.model,
                                     discriminator=discriminator.model)
    # Start Training
    for epoch in range(EPOCHS):
        print("\nEPOCH:", epoch)
        start = time.time()

        for batch in range(dataset.train_data.shape[0] // BATCH_SIZE):
            image_batch = dataset.train_data[np.random.randint(0, dataset.train_data.shape[0], size=BATCH_SIZE)]
            train_step(image_batch, generator, discriminator)

        # Save the model every 15 epochs
        if (epoch + 1) % 1 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        generate_and_save_images(generator.model,
                                 epoch,
                                 SEED, dataset)

    # Generate after the final epoch
    generate_and_save_images(generator.model,
                             EPOCHS,
                             SEED, dataset)


if __name__ == "__main__":
    print("Program Start!")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("\n----Hyperparameters----")
    print("Epochs:", EPOCHS, "\nBatch size:", BATCH_SIZE, "\nDataset:", DATASET)

    print("\nLoading Dataset and Models...")
    if DATASET == "MNIST":
        dataset, generator, discriminator = MNIST(), MNISTGenerator(), MNISTDiscriminator()
    elif DATASET == "CIFAR10":
        dataset, generator, discriminator = CIFAR(), CIFARGenerator(), CIFARDiscriminator()
    else:
        print("DATASET constant is not set to a valid dataset. Set to MNIST or CIFAR10.")
        print("Program End!")
        pass
    print("Dataset", DATASET, "and Models Loaded!")
    # image = dataset.train_data[0]
    # print(image, image.shape)
    # image = (image + 1)/2
    # print(image, image.shape)
    # plt.imshow(image, interpolation='nearest')
    # plt.show()

    print("Training Start!")
    train(dataset, generator, discriminator)
