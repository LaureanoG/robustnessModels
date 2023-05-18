# Imports
import os, sys
import tensorflow as tf
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
import matplotlib as mpl
import matplotlib.pyplot as plt

from art.attacks.evasion import FastGradientMethod, CarliniLInfMethod, ProjectedGradientDescentTensorFlowV2
from art.estimators.classification import TensorFlowV2Classifier
from art.utils import load_dataset

sys.path.append(os.path.abspath(os.path.join('..', '')))
from Attacks.attacks_helper import *
from generator_v2 import *
from discriminator_v2 import *

# Constants
DATASET = "CIFAR10"

# Start of Main Function
print("Program Start!")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.__version__[0] != '2':
    raise ImportError('This notebook requires TensorFlow v2.')
cifar, mnist = CIFAR(), MNIST()
if DATASET == "CIFAR10":
    testModel = CIFARDiscriminator().model
    testModel.load_weights("DiscModelCIFAR10.h5")
    # ogModel.fit(cifar.train_data, cifar.train_labels, epochs = 25)
    # ogModel.save("og_model_1.h5")
    ogModel = CIFARModel("../Models/CIFAR10").model
else:
    testModel, ogModel = None, None
print("Finished Loading Models and Datasets")


# Runs on CIFAR-10 on OG Model
print("********************************\n* CIFAR 10 - Before GAT Training *\n********************************")
print("--Before Attacking--")
test_data_results = ogModel.evaluate(cifar.test_data, cifar.test_labels)
print('Accuracy on Validation Data: {}%\tLoss on Validation Data: {}'.format(test_data_results[1], test_data_results[0]))

print("\n--Fast Sign Gradient Method--")
CIFAR_10_classifier = TensorFlowV2Classifier(model=ogModel, clip_values=(0, 1), nb_classes=10, \
                                          input_shape=(32, 32, 3), channels_first=False, \
                                          loss_object=attacks_helper.fn)


attack_fgsm = FastGradientMethod(estimator=CIFAR_10_classifier, norm="inf", eps=8, minimal=True)  # Generates Adversarial Data
x_test_adv = attack_fgsm.generate(x=cifar.test_data[:1000])

y_test_pred = np.argmax(tf.nn.softmax(ogModel(x_test_adv)), axis=1)
adv_data_result = ogModel.evaluate(x_test_adv, cifar.test_labels[:1000])  # Runs Attack
perturbation = np.mean(np.abs((x_test_adv - cifar.test_data[:1000])))

plt.matshow(x_test_adv[0, :, :, 0])  # Shows a perbutation
plt.savefig('FGSM_CIFAR_UNTRAINED.png')  # Saves a perbutation  # Reports results
print('Accuracy on Adversarial Data: {}%'.format(adv_data_result))
print('Average perturbation: {}'.format(perturbation))

# Runs cifar10 on adv trained model
print("********************************\n* CIFAR 10 - After GAT Training *\n********************************")
print("--Before Attacking--")
test_data_results = testModel.evaluate(cifar.test_data, cifar.test_labels)
print('Accuracy on Validation Data: {}%\tLoss on Validation Data: {}'.format(test_data_results[1], test_data_results[0]))

print("\n--Fast Sign Gradient Method--")
CIFAR_10_classifier = TensorFlowV2Classifier(model=testModel, clip_values=(0, 1), nb_classes=10, \
                                          input_shape=(32, 32, 3), channels_first=False, \
                                          loss_object=tf.keras.losses.CategoricalCrossentropy())


attack_fgsm = FastGradientMethod(estimator=CIFAR_10_classifier, norm="inf", eps = 8, minimal=True)  # Generates Adversarial Data
x_test_adv = attack_fgsm.generate(x=cifar.test_data[:1000])

y_test_pred = np.argmax(tf.nn.softmax(testModel(x_test_adv)), axis=1)
adv_data_result = testModel.evaluate(x_test_adv, cifar.test_labels[:1000])  # Runs Attack
perturbation = np.mean(np.abs((x_test_adv - cifar.test_data[:1000])))

plt.matshow(x_test_adv[0, :, :, 0])  # Shows a perbutation
plt.savefig('FGSM_CIFAR_TRAINED.png')  # Saves a perbutation  # Reports results
print('Accuracy on Adversarial Data: {}%'.format(adv_data_result))
print('Average perturbation: {}'.format(perturbation))