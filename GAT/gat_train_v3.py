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
EPOCHS = 50
BATCH_SIZE = 128
DATASET = "CIFAR10"
K_STEPS = 1
COLOR_CHANNEL_NUM = 3
CG = 0.03
ALPHA = 0.5


# Defines GAN model
def get_gan(discriminator, generator):
    discriminator.model.trainable = False
    generator.model.trainable = True

    images_input = tf.keras.Input(shape=(32, 32, 3))
    gradient_input = tf.keras.Input(shape=(32, 32, 3))
    gen_output = generator.model(gradient_input)
    adversarial_input = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([gen_output, images_input])

    gan_output = discriminator.model(adversarial_input)
    gan = tf.keras.Model(inputs=[images_input, gradient_input], outputs=[gan_output, gen_output])

    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    gan.compile(loss=GAN_loss, optimizer=opt)
    return gan


# Defines GAN loss funct
def GAN_loss(fake_pred, labels, perturbation):
    fake_pred = tf.cast(fake_pred, dtype="float64")
    real = tf.reduce_sum(labels * fake_pred, 1)
    other = tf.reduce_max((1 - labels) * fake_pred - (labels * 10000), 1)
    zero = tf.cast(0, dtype="float64")
    adv_loss = tf.reduce_sum(tf.maximum(zero, real - other))

    perturb_loss = tf.cast(tf.math.square(tf.norm(perturbation)), dtype="float64")
    return adv_loss + CG * perturb_loss


# Get image gradients for generator input
def get_gradients(images, discriminator):
    discriminator.trainable = False
    # Compute input gradient image with respect to output
    x_tensor = tf.convert_to_tensor(images, dtype=tf.float32)
    with tf.GradientTape() as t:
        t.watch(x_tensor)
        output = discriminator.model(x_tensor)
    return t.gradient(output, x_tensor)


def generate_and_save_images(model, epoch, dataset):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = tf.clip_by_value(model(dataset.test_data[:49], training=False), 0, 1)

    fig = plt.figure(figsize=(4, 4))

    # Saves Adversarial Examples
    print("Saving Adversarial Examples...")
    for i in range(predictions.shape[0]):
        plt.subplot(7, 7, i + 1)
        # print(predictions[i, :, :, :], predictions[i, :, :, :].shape)
        adversarial = dataset.test_data[i] + predictions[i, :, :, :]
        plt.imshow((adversarial + 1) / 2, interpolation="bicubic")
        plt.axis('off')

    plt.savefig('Images_GAT/{}_image_at_epoch_{:04d}.png'.format(DATASET, epoch))


    # Saves Perturbations
    print("Saving Perturbations...")

    for i in range(predictions.shape[0]):
        plt.subplot(7, 7, i + 1)
        # print(predictions[i, :, :, :], predictions[i, :, :, :].shape)
        plt.imshow((predictions[i, :, :, :] + 1) / 2, interpolation="None")
        plt.axis('off')

    plt.savefig('Images_GAT/{}_perturbation_at_epoch_{:04d}.png'.format(DATASET, epoch))
    plt.close()


@tf.function
def train_step_gen(images, generator, discriminator, imageLabels, gan):
    discriminator.model.trainable = False
    with tf.GradientTape() as tape:
        gradients = get_gradients(images, discriminator)
        gan_predict, gen_output = gan([images, gradients], training=True)
        gen_loss = GAN_loss(gan_predict,  imageLabels, gen_output)

    grads = tape.gradient(gen_loss, gan.trainable_variables)
    generator.optimizer.apply_gradients(zip(grads, gan.trainable_variables))
    return gen_loss


def train_step_disc(images, generator, discriminator, imageLabels):
    # Generates Adversarial Images
    gradients = get_gradients(images, discriminator)
    generated_images = generator.model.predict(gradients)
    adversarial_images = generated_images + images

    discriminator.model.trainable = True
    # discriminator.model.train_on_batch(images, imageLabels)
    # discriminator.model.train_on_batch(adversarial_images, imageLabels)
    cross_entropy = tf.keras.losses.CategoricalCrossentropy()

    with tf.GradientTape() as tape:
          real_output = discriminator.model(images, training=True)
          fake_output = discriminator.model(adversarial_images, training=True)
          disc_loss = ALPHA * cross_entropy(imageLabels, real_output) + (1 - ALPHA) * cross_entropy(imageLabels, fake_output)

    grads = tape.gradient(disc_loss, discriminator.model.trainable_variables)
    discriminator.optimizer.apply_gradients(zip(grads, discriminator.model.trainable_variables))


def train(dataset, generator, discriminator, gan):
    # Initialize Checkpoints
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator.optimizer,
                                     discriminator_optimizer=discriminator.optimizer,
                                     generator=generator.model,
                                     discriminator=discriminator.model)

    # Start Training
    for epoch in range(EPOCHS):
        print("\nEPOCH:", epoch)
        start = time.time()

        gen_losses = []
        for i in range(0, K_STEPS):
            print("Generator is training...")
            for batch in range(dataset.train_data.shape[0] // BATCH_SIZE):
                random_val = np.random.randint(0, dataset.train_data.shape[0], size=BATCH_SIZE)
                image_batch, image_batch_label = dataset.train_data[random_val], dataset.train_labels[random_val]
                gen_loss = train_step_gen(image_batch, generator, discriminator, image_batch_label, gan)
                gen_losses.append(gen_loss)

        print("Discriminator is training...")
        for batch in range(dataset.train_data.shape[0] // BATCH_SIZE):
            random_val = np.random.randint(0, dataset.train_data.shape[0], size=BATCH_SIZE)
            image_batch, image_batch_label = dataset.train_data[random_val], dataset.train_labels[random_val]
            train_step_disc(image_batch, generator, discriminator, image_batch_label)

        # Save the model every 15 epochs
        if (epoch + 1) % 1 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        # Prints metrics
        test_labels = dataset.test_labels[:500]
        test_data = dataset.test_data[:500]
        real_acc = discriminator.model.evaluate(test_data, test_labels)

        generated_images = generator.model.predict(test_data)
        adversarial_images = generated_images + test_data
        fake_acc = discriminator.model.evaluate(adversarial_images, test_labels)

        print("Discriminator Real Accuracy:", real_acc[1])
        print("Discriminator Fake Accuracy:", fake_acc[1])
        print("GAN Loss:", GAN_loss(discriminator.model(adversarial_images), test_labels, generated_images))
        print("Perturbation:", np.mean(np.abs((adversarial_images - test_data))))
        print('Time for epoch {} is {} sec'.format(epoch, time.time() - start))
        generate_and_save_images(generator.model,
                                 epoch,
                                 dataset)

    # Save Final Models
    print("Saving Models...")
    discriminator.model.save("DiscModel"+str(DATASET)+".h5")
    generator.model.save("GenModel"+str(DATASET)+".h5")
    print("Models saved!")


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
    gan = get_gan(discriminator, generator)
    print("Dataset", DATASET, "and Models Loaded!")

    print("Training Start!")
    train(dataset, generator, discriminator, gan)
    print("Training End!")
