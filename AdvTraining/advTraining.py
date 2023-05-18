import numpy as np
import sys, os, time
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data

sys.path.append(os.path.abspath(os.path.join('..', '')))

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import TensorFlowV2Classifier
from TraineeModels.setup_mnist import *
from TraineeModels.setup_cifar import *

# Constants
EPSILON = 8
DATASET = "CIFAR10"
EPOCHS = 50
BATCH_SIZE = 128


def get_attacks(model, input_image):
    if dataset == "CIFAR10":
        model_classifier = TensorFlowV2Classifier(model=model, clip_values=(0, 1), nb_classes=10, \
                                                     input_shape=(32, 32, 3), channels_first=False, \
                                                     loss_object=tf.keras.losses.CategoricalCrossentropy())
    else:
        model_classifier = TensorFlowV2Classifier(model=model, clip_values=(0, 1), nb_classes=10, \
                                                  input_shape=(28, 28, 1), channels_first=False, \
                                                  loss_object=tf.keras.losses.CategoricalCrossentropy())

    attack_fgsm = FastGradientMethod(estimator=model_classifier, norm="inf", eps=EPSILON, minimal=True)
    return attack_fgsm.generate(x=input_image)


def train(dataset, model):
    # Start Training
    for epoch in range(EPOCHS):
        print("\nEPOCH:", epoch)
        start = time.time()

        print("Model is training...")
        for batch in range(dataset.train_data.shape[0] // BATCH_SIZE):
            random_val = np.random.randint(0, dataset.train_data.shape[0], size=BATCH_SIZE)
            image_batch, image_batch_label = dataset.train_data[random_val], dataset.train_labels[random_val]
            reg_loss = model.train_on_batch(image_batch, image_batch_label)
            adv_loss = model.train_on_batch(get_attacks(model, image_batch), image_batch_label)

        # Prints metrics
        print("Calculating Metrics...")
        test_data = dataset.test_data[:1000]
        test_labels = dataset.test_labels[:1000]
        test_attacks = get_attacks(model, test_data)

        model.evaluate(test_data, test_labels)
        model.evaluate(test_attacks, test_labels)
        print('Time for epoch {} is {} sec'.format(epoch, time.time() - start))

    # Save Final Models
    print("Saving Models...")
    model.save("Adv_FGSM_"+str(DATASET)+".h5")
    print("Models saved!")


if __name__ == "__main__":
    print("Program Start!")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("\n----Hyperparameters----")
    print("Epochs:", EPOCHS, "\nBatch size:", BATCH_SIZE, "\nDataset:", DATASET)

    print("\nLoading Dataset and Models...")
    if DATASET == "MNIST":
        dataset, model= MNIST(), MNISTModel(None).model
    elif DATASET == "CIFAR10":
        dataset, model = CIFAR(), CIFARModel(None).model
    else:
        print("DATASET constant is not set to a valid dataset. Set to MNIST or CIFAR10.")
        print("Program End!")
        pass
    print("Dataset", DATASET, "and Models Loaded!")

    print("Training Start!")
    train(dataset, model)
    print("Training End!")
