# Imports
import tensorflow as tf
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution
import matplotlib as mpl
import matplotlib.pyplot as plt

from art.attacks.evasion import FastGradientMethod, CarliniLInfMethod, ProjectedGradientDescentTensorFlowV2
from art.estimators.classification import TensorFlowV2Classifier
from art.utils import load_dataset
from attacks_helper import *

# Start of Main Function
print("Program Start!")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.__version__[0] != '2':
    raise ImportError('This notebook requires TensorFlow v2.')
cifarModel, mnistModel = loadModels()
cifar, mnist = CIFAR(), MNIST()
print("Finished Loading Models and Datasets")

# Attacks runs on MNIST
# print("******************************\n* MNIST - Before GAT Training *\n******************************")
# print("--Before Attacking--")
# test_data_results = mnistModel.evaluate(mnist.test_data, mnist.test_labels)
# print('Accuracy on Validation Data: {}%\tLoss on Validation Data: {}'.format(test_data_results[1], test_data_results[0]))
#
# print("\n--Fast Sign Gradient Method--")
# MNIST_classifier = TensorFlowV2Classifier(model=mnistModel, clip_values=(0, 1), nb_classes=10, \
#                                           input_shape=(28, 28, 1), channels_first=False, \
#                                           loss_object=fn)
#
# attack_fgsm = FastGradientMethod(estimator=MNIST_classifier, norm="inf", eps = 0.3, minimal=True)  # Generates Adversarial Data
# x_test_adv = attack_fgsm.generate(x=mnist.test_data[:1000])
# y_test_pred = np.argmax(tf.nn.softmax(mnistModel(x_test_adv)), axis=1)
#
# adv_data_result = mnistModel.evaluate(x_test_adv[:1000], mnist.test_labels[:1000])  # Runs Attack
# perturbation = np.mean(np.abs((x_test_adv - mnist.test_data[:1000])))
#
# plt.imshow(x_test_adv[0, :, :, 0], cmap='gray')  # Shows a perbutation
# plt.savefig('FGSM_MNIST.png')  # Saves a perbutation
# print('Accuracy on Adversarial Data: {}%'.format(adv_data_result))  # Reports results
# print('Average perturbation: {:4.2f}'.format(perturbation))
#
# print("\n--Carlini & Wagner--")
# attack_cw = CarliniLInfMethod(classifier=MNIST_classifier,
#                                max_iter=1000,
#                                learning_rate=0.005)
# x_test_adv = attack_cw.generate(x=mnist.test_data[:1000])
#
# accuracy_test_adv = mnistModel.evaluate(x_test_adv, mnist.test_labels[:1000])
# perturbation = np.mean(np.abs((x_test_adv - mnist.test_data[:1000])))
# print('Accuracy on Adversarial Data: {:4.2f}%'.format(accuracy_test_adv[1] * 100))
# print('Average perturbation: {:4.2f}'.format(perturbation))
# plt.matshow(x_test_adv[0, :, :, 0])  # Shows a perbutation
# plt.savefig('C&W_MNIST.png')  # Saves perbutation
#
# print("\n--PGD--")
# attack_pgd = ProjectedGradientDescentTensorFlowV2(estimator=MNIST_classifier, norm=np.inf, max_iter= 1000, eps = 0.3)
# x_test_adv = attack_pgd.generate(x=mnist.test_data[:1000])
#
# y_test_pred = np.argmax(tf.nn.softmax(mnistModel(x_test_adv)), axis=1)
# adv_data_result = mnistModel.evaluate(x_test_adv, mnist.test_labels[:1000])  # Runs Attack
# perturbation = np.mean(np.abs((x_test_adv - mnist.test_data[:1000])))
#
# plt.matshow(x_test_adv[0, :, :, 0])  # Shows a perbutation
# plt.savefig('FGSM_MNIST.png')  # Saves a perbutation # Reports results
# print('Accuracy on Adversarial Data: {}%'.format(adv_data_result))
# print('Average perturbation: {:4.2f}'.format(perturbation))

# # Runs on CIFAR-10
print("***************************\n* CIFAR 10 - Before GAT Training *\n***************************")
print("--Before Attacking--")
test_data_results = cifarModel.evaluate(cifar.test_data, cifar.test_labels)
print('Accuracy on Validation Data: {}%\tLoss on Validation Data: {}'.format(test_data_results[1], test_data_results[0]))

print("\n--Fast Sign Gradient Method--")
CIFAR_10_classifier = TensorFlowV2Classifier(model=cifarModel, clip_values=(0, 1), nb_classes=10, \
                                          input_shape=(32, 32, 3), channels_first=False, \
                                          loss_object=fn)


attack_fgsm = FastGradientMethod(estimator=CIFAR_10_classifier, norm="inf", eps = 8, minimal=True)  # Generates Adversarial Data
x_test_adv = attack_fgsm.generate(x=cifar.test_data[:1000])

y_test_pred = np.argmax(tf.nn.softmax(cifarModel(x_test_adv)), axis=1)
adv_data_result = cifarModel.evaluate(x_test_adv, cifar.test_labels[:1000])  # Runs Attack
perturbation = np.mean(np.abs((x_test_adv - cifar.test_data[:1000])))

plt.matshow(x_test_adv[0, :, :, 0])  # Shows a perbutation
plt.savefig('FGSM_CIFAR.png')  # Saves a perbutation  # Reports results
print('Accuracy on Adversarial Data: {}%'.format(adv_data_result))
print('Average perturbation: {}'.format(perturbation))

# print("\n--Carlini & Wagner--")
# attack_cw = CarliniLInfMethod(classifier=CIFAR_10_classifier,
#                               max_iter=1000,
#                               learning_rate=0.005)
#
# x_test_adv = attack_cw.generate(x=cifar.test_data[:1000])
#
# accuracy_test_adv = mnistModel.evaluate(x_test_adv, cifar.test_labels[:1000])
# perturbation = np.mean(np.abs((x_test_adv - cifar.test_data[:1000])))
# print('Accuracy on Adversarial Data: {:4.2f}%'.format(accuracy_test_adv[1] * 100))
# print('Average perturbation: {:4.2f}'.format(perturbation))
# plt.matshow(x_test_adv[0, :, :, 0])  # Shows a perbutation
# plt.savefig('C&W_CIFAR.png')  # Saves perbutation

print("\n--PGD--")
attack_pgd = ProjectedGradientDescentTensorFlowV2(estimator=CIFAR_10_classifier, norm=np.inf, max_iter=1000, eps = 8)
x_test_adv = attack_pgd.generate(x=cifar.test_data[:1000])

y_test_pred = np.argmax(tf.nn.softmax(cifarModel(x_test_adv)), axis=1)
adv_data_results = np.sum(y_test_pred == cifar.test_labels[:1000]) / cifar.test_labels[:1000].shape[0]
adv_data_result = cifarModel.evaluate(x_test_adv, cifar.test_labels[:1000])  # Runs Attack
perturbation = np.mean(np.abs((x_test_adv - cifar.test_data[:1000])))

plt.matshow(x_test_adv[0, :, :, 0])  # Shows a perbutation
plt.savefig('PGD_CIFAR.png')  # Saves a perbutation
print('Accuracy on Adversarial Data: {}%'.format(adv_data_results))  # Reports results
print('Accuracy on Adversarial Data: {}%'.format(adv_data_result))
print('Average perturbation: {:4.2f}'.format(perturbation))

