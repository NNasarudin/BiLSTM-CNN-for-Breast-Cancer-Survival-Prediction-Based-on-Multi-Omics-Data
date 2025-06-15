# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03 22:30:17 2023

@author: Athirah
"""
import keras
import keras.backend as K
from keras.layers import Input, Dropout, Flatten, Dense, MaxPooling1D, Activation
from keras.layers import multiply, concatenate
# from keras.layers.merge import concatenate
from sklearn.model_selection import StratifiedKFold, train_test_split
# demonstrate data normalization with sklearn
from sklearn.preprocessing import MinMaxScaler
import numpy

from sklearn.metrics import roc_curve, auc
from keras.models import Model
from keras.utils import plot_model
from keras.regularizers import l2
from keras.layers.convolutional import Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Multiply
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import LSTM
from keras import initializers, regularizers
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, \
    BatchNormalization
import matplotlib.pyplot as plt
import tensorflow as tf

num_of_filters = 25
epochs = 20
Sp_value = 0.95
acc_cvscores = []
Pr_cvscores = []
Sn_cvscores = []
Mcc_cvscores = []
path = 'C:/Users/athir/Downloads/DL_EHR_Prediction/SiGaAtCNNstackedRF-master/Data/tcgaExpOutput.csv'  # Change the path to your local system


def shows_result(path, arr):
    with open(path, 'w') as f:
        for item_clinical in arr:
            for elem in item_clinical:
                f.write(str(elem) + ',')
            f.write('\n')


def nlrelu(t, label):
    if label == 'nlrelu':
        return tf.log(tf.nn.relu(t) + 1.)
    elif label == 'selu':
        return tf.nn.selu(t)


'''
code to calculate sensitivity at given specificity value
'''


def sensitivity_at_specificity(specificity, **kwargs):
    def Sn(labels, predictions):
        # any tensorflow metric
        value, update_op = K.tf.metrics.sensitivity_at_specificity(labels, predictions, specificity, **kwargs)

        # find all variables created for this metric
        metric_vars = [i for i in K.tf.local_variables() if 'sensitivity_at_specificity' in i.name.split('/')[2]]

        # Add metric variables to GLOBAL_VARIABLES collection.
        # They will be initialized for new session.
        for v in metric_vars:
            K.tf.add_to_collection(K.tf.GraphKeys.GLOBAL_VARIABLES, v)

        # force to update metric values
        with K.tf.control_dependencies([update_op]):
            value = K.tf.identity(value)
            return value

    return Sn


# Define a custom layer for self-attention
class SelfAttentionLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SelfAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Calculate attention scores
        scores = tf.matmul(inputs, tf.transpose(inputs, perm=[0, 2, 1]))
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # Apply attention weights to the inputs
        attended_inputs = Multiply()([inputs, attention_weights])

        return attended_inputs


# fix random seed for reproducibility
numpy.random.seed(1)

'''load METABRIC Clinical dataset'''
#dataset_clinical = numpy.loadtxt(
   # "C:/Users/athir/Downloads/DL_EHR_Prediction/SiGaAtCNNstackedRF-master/Data/METABRIC/METABRIC_clinical_1980.txt")  # Change the path to your local system
'''split into input (X) and output (Y) variables'''
#X_clinical = dataset_clinical[:, 0:25]
#Y_clinical = dataset_clinical[:, 25]

''' load TCGA Clinical dataset '''
X_clinical = numpy.loadtxt("C:/Users/athir/Downloads/DL_EHR_Prediction/SiGaAtCNNstackedRF-master/Data/TCGA/tcga_exp_final.csv", delimiter=",") # Change the path to your local system
Y_clinical = numpy.loadtxt("C:/Users/athir/Downloads/DL_EHR_Prediction/SiGaAtCNNstackedRF-master/Data/TCGA/5yearCutOff.txt")


# 10 fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
cvscores = []
i = 1

for train_index, test_index in kfold.split(X_clinical, Y_clinical):
    print(i, "th Fold *****************************************")
    i = i + 1
    # Spliting the clinical data set into training and testing
    x_train_clinical, x_test_clinical = X_clinical[train_index], X_clinical[test_index]
    y_train_clinical, y_test_clinical = Y_clinical[train_index], Y_clinical[test_index]
    x_train_clinical = numpy.expand_dims(x_train_clinical, axis=2)
    x_test_clinical = numpy.expand_dims(x_test_clinical, axis=2)

    # first Clinical CNN Model***********************************************************
    init = initializers.glorot_normal(seed=1)
    bias_init = keras.initializers.Constant(value=0.1)
    #main_input_clinical = Input(shape=(25, 1), name='Input')  # for metabric data
    main_input_clinical = Input(shape=(400,1),name='Input')# for TCGA data

    conv1 = Conv1D(filters=32, kernel_size=15, strides=2, activation='relu', padding='same', name='Conv1D',
                   kernel_initializer=init, bias_initializer=bias_init)(main_input_clinical)
    conv2 = Conv1D(filters=32, kernel_size=15, strides=2, activation='relu', padding='same', name='Conv1D2',
                   kernel_initializer=init, bias_initializer=bias_init)(conv1)
    flat1 = Flatten(name='Flatten')(conv2)
    dense1 = Dense(150, activation='tanh', name='dense1', kernel_initializer=init, bias_initializer=bias_init)(flat1)
    output = Dense(1, activation='sigmoid', name='output', activity_regularizer=regularizers.l2(0.01),
                   kernel_initializer=init, bias_initializer=bias_init)(dense1)


    model = Model(inputs=main_input_clinical, outputs=output)
    plot_model(model,
               to_file='C:/Users/athir/Downloads/DL_EHR_Prediction/SiGaAtCNNstackedRF-master/Data/clinical_gated_attention.png')  # Change the path to your local system

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=[sensitivity_at_specificity(Sp_value)])

    x_train1, x_val1, y_train1, y_val1 = train_test_split(x_train_clinical, y_train_clinical, test_size=0.2,
                                                          stratify=y_train_clinical)
    model.fit(x_train1, y_train1, epochs=epochs, batch_size=8, validation_data=(x_val1, y_val1))

    scores = model.evaluate(x_test_clinical, y_test_clinical, verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    Sn_cvscores.append(scores[1] * 100)
print("Sensitivity = %.2f%% (+/- %.2f%%)\n" % (numpy.mean(Sn_cvscores), numpy.std(Sn_cvscores)))

X_clinical = numpy.expand_dims(X_clinical, axis=2)
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("dense1").output)
intermediate_output = intermediate_layer_model.predict(X_clinical)
shows_result(path, intermediate_output)

y_pred = model.predict(X_clinical)
fpr, tpr, thresholds = roc_curve(Y_clinical, y_pred, pos_label=1)


def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])


for threshold in numpy.arange(0, 1, 0.05):
    print('********* Threshold = ', threshold, ' ************')
    evaluate_threshold(threshold)

roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'r', label='BiLSTM_CNN-CLN = %0.3f' % roc_auc)
plt.xlabel('1-Sp (False Positive Rate)')
plt.ylabel('Sn (True Positive Rate)')
plt.title('Receiver Operating Characteristics')
plt.legend()
plt.show()

