import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
import pandas as pd
import math

np.random.seed(0)  # fix random functions for evaluatory purposes

LEARN_TF = False

# To write a TensorFlow program based on pre-made Estimators, you must perform the following tasks:
# Create one or more input functions.
# Define the model's feature columns.
# Instantiate an Estimator, specifying the feature columns and various hyperparameters.
# Call one or more methods on the Estimator object, passing the appropriate input function as the source of the data.

CSV_COLUMNS = 'LIMIT_BAL	SEX	EDUCATION	MARRIAGE	AGE	PAY_0	PAY_2	PAY_3	PAY_4	PAY_5	PAY_6	BILL_AMT1	BILL_AMT2	BILL_AMT3	BILL_AMT4	BILL_AMT5	BILL_AMT6	PAY_AMT1	PAY_AMT2	PAY_AMT3	PAY_AMT4	PAY_AMT5	PAY_AMT6	DEFAULT'.split()
CSV_COLUMN_DEFAULTS = [[0]] * len(CSV_COLUMNS) # all numeric data

# ======== Hyperparameters =============
batch_size = 32
num_features = 24 # set equal to num features of data
epochs = 20




#The Dataset API can handle a lot of common cases for you.
# For example, using the Dataset API, you can easily read in records from a large collection of files in parallel and join them into a single stream.

def create_feature_cols(features):
    my_feature_cols = []
    for key in features.keys():
        my_feature_cols.append(tf.feature_column.numeric_column(key = key))
    return my_feature_cols

def create_feature_dict(feature_values):
    features = {}
    for i in range(len(CSV_COLUMNS) - 1):
        features[CSV_COLUMNS[i]] = feature_values[CSV_COLUMNS[i]]
    return features

def input_function(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))  # create a dataset from provided features and labels

    dataset = dataset.shuffle(buffer_size= 1000).repeat().batch(batch_size) # dataset.shuffle shuffles the data after every epoch. NOTE: VERY important to avoid overfitting
    # .repeat() is the number of times we want to iterate over the dataset. default value = infinity, which is commonly used so you can iterate over it in a for loop
    # buffersize is the number over which the next element will be uniformly chosen from

    return dataset.make_one_shot_iterator().get_next()   # create an iterator for our dataset

def get_data(training_perc = 0.6, val_perc = 0.2, test_perc = 0.2):   #NOTE: don't need to randomize data here
    assert training_perc + val_perc + test_perc == 1

    # ORGANIZING DATA IN PANDAS
    raw_data = pd.read_csv(r'H:\Summer Research 2018\Credit Default Forecasting\UCI_Credit_Card.csv')
    # raw_data.dropna()     # drop all na data points for this experiment
    # assert not np.any(np.isnan(np.array(raw_data)))  # double check no nan data
    num_data = raw_data.shape[0]        # number of total data points


    # Normalize the data (choose 1 or 2):
    # 1. Divide each column by the max value of the column. THis will constrain input values to [0,1]
    # 2. Subtract mean and divide by standard deviation for each col. This generates a zero mean and variance of one for each feature value
    for col_name in CSV_COLUMNS[0:-1]:
        col_mean = np.mean(raw_data[col_name]) ; col_std = np.std(raw_data[col_name])
        raw_data[col_name] = raw_data[col_name].map(lambda x: (x - col_mean) / col_std)   # OPTION 2.

    random_indices = np.random.permutation(num_data)
    end_train, end_val = int(num_data * training_perc), int(num_data * training_perc) + int(num_data * val_perc)
    training_set, training_labels = raw_data.iloc[random_indices[0:end_train], 0:-1],  raw_data.iloc[random_indices[0:end_train], -1]
    val_set, val_labels = raw_data.iloc[random_indices[end_train:end_val], 0:-1],  raw_data.iloc[random_indices[end_train:end_val], -1]
    test_set, test_labels = raw_data.iloc[random_indices[end_val:], 0:-1],  raw_data.iloc[random_indices[end_val:], -1]
    if LEARN_TF:
        print("Training:", training_set.shape, training_labels.shape)
        print("Val:", val_set.shape, val_labels.shape)
        print("Test:", test_set.shape, test_labels.shape)
        print(training_labels.iloc[0:10])

    return training_set, val_set, test_set, training_labels, val_labels, test_labels

# Parameters
# learning_rate = 0.0000001
learning_rate = .00001
num_steps = 1000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
# num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# Define the neural network
def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['records']
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    logits = neural_net(features)

    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs



def logistic_regression():
    X = tf.placeholder(tf.float32, shape = (num_features, None))
    W = tf.Variable(tf.random_normal((1, num_features)), name = "W")
    b = tf.Variable(tf.zeros(None, 1))
    y = tf.placeholder(tf.int, shape = (1, None))
    y_hat = tf.sigmoid(tf.matmul(W, X))
    return y_hat

def easy_logistic_regression(Data,labels):
    X = tf.placeholder(tf.float32, shape = (num_features, None))
    W = tf.Variable(tf.random_normal((1, num_features)), name = "W")
    b = tf.Variable(tf.zeros(None, 1))
    y = tf.placeholder(tf.int, shape = (1, None))
    y_hat = tf.sigmoid(tf.matmul(W, X))

    log_reg = tf.nn.softmax(tf.matmul(W,X) + b)
    num_batches = Data.shape[0] // batch_size

    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)  # initialize Variables
    #     for epoch in range(epochs):
    #
    #
    #         for batch in range(num_batches):
    #                 tf.train.batch(allow_smaller_final_batch= True)





if __name__ == '__main__':
    data = get_data()

    # y_hat = logistic_regression()
    # print(y_hat)
    training_set, val_set, test_set, training_labels, val_labels, test_labels = get_data()
    print(training_set.shape)

























# FOR TF:
# training_features = create_feature_dict(training_set)  # use training set to create features dictionary
# my_feature_cols = create_feature_cols(training_features)





#  ================================ KERAS ==================================================================
#
# def precision_recall(predictions, labels, threshold = 0.5):
#     assert len(predictions) == len(labels)
#     labels = np.array(labels)
#     # Precision = (TP)  / (TP + FP)
#     # Recall = (TP) / (TP + FN)
#     # print("shapes of preds & labels", predictions.shape, labels.shape)
#     predictions[predictions < threshold] = 0
#     predictions[predictions >= threshold] = 0
#     TP = np.sum(labels)  # since it's binary, 1 = True, 0 = False, number of defaults
#     FP = np.sum([1 for ex in range(predictions.shape[0]) if predictions[ex,0] == 1 and labels[ex] == 0]) # num of times we predicted 1, but answer was 0
#     FN = np.sum([1 for ex in range(predictions.shape[0]) if predictions[ex,0] == 0 and labels[ex] == 1]) # num of times we predicted 0, but the answer was 1
#     precision = TP / (TP + FP)
#     recall = TP / (TP + FN)
#     F1_score = 2 * ((precision * recall ) / (precision + recall))
#     print(type(precision), type(recall), type(F1_score))
#     return precision, recall, F1_score
#
# def show_data_distribution(training, val, test):
#     full_data_labels = np.concatenate((training, val, test))
#     print("Percentage of defaults in population: " + str(round(np.sum(full_data_labels) / len(full_data_labels),3)))
#     print("Percentage of defaults in training set: " + str(round(np.sum(training_labels) / len(training_labels),3)))
#     print("Percentage of defaults in validation set: " + str(round(np.sum(val_labels) / len(val_labels),3)))
#     print("Percentage of defaults in test set: " + str(round(np.sum(test_labels) / len(test_labels),3)))
#
# def accuracy(pred, labels, threshold = 0.5):
#     pred[pred < threshold] = 0
#     pred[pred >= threshold] = 1
#     acc = (np.sum(pred[:, 0] == labels) / pred.shape[0], 3)
#     return acc
#
# from keras.models import Sequential
# from keras.layers import Dense, Activation
#
# training_set, val_set, test_set, training_labels, val_labels, test_labels = get_data()  # obtain, randomize and divide data into appropriate sets
#
# # Layer sizes
# h1 = 15  # number of units in hidden layer 1
# h2 = 10  # num units in layer 2
# h3 = 10
# h4 = 5
# epochs = 20
# batch = 128  # batch size
#
# model = Sequential([
#     Dense(h1, input_shape=(training_set.shape[1],)),
#     Activation('relu'),
#     Dense(1),
#     Activation('sigmoid'),
# ])
#
# # show_data_distribution(training_labels, val_labels, test_labels)
#
# model.compile(loss='binary_crossentropy', optimizer= 'rmsprop', metrics=['accuracy'])
#
# model.fit(training_set, training_labels, epochs=epochs, batch_size=batch)
# train_loss, train_accuracy = model.evaluate(training_set, training_labels, batch_size=128)
# val_score = model.evaluate(val_set, val_labels, batch_size=128)
#
# print("Training loss:", round(train_loss,3), "Training accuracy:", round(train_accuracy,3))
# print("Validation loss: " + str(round(val_score[0],3)) + " Validation accuracy: " +str(round(val_score[1],3)))  # These metrics might be different from mine because they averaged over batches???
#
# training_predictions = model.predict(training_set, batch_size=128)
# val_predictions = model.predict(val_set, batch_size=128)
#
# precision, recall, F1_score = precision_recall(training_predictions, training_labels)
# print("Precision: ", round(precision,3), "Recall:", round(recall,3), "F1score:", round(F1_score,3))
#
#
# #  ---------- Plotting precision-recall curve -----------
# from sklearn.metrics import precision_recall_curve
# import matplotlib.pyplot as plt
# from sklearn.metrics import average_precision_score
#
# # val_predictions = val_predictions[:,0]
#
# print(val_labels.shape, val_predictions.shape)
# average_precision = average_precision_score(training_labels, training_predictions)
# precision, recall, _ = precision_recall_curve(training_labels, training_predictions)
#
# print("SKLEARN precision:", len(precision), "Recall:", len(recall))
#
# plt.step(recall, precision, color='b', alpha=0.2, where='post')  # alpha controls opaquenes, where controls where the steps are placed
# plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
#
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
# plt.show()
#------------------------------------------



# TODO: Get tensorboard up and evlauate metrics

# TODO: Learn what sparse tensors are for and what sparse features are

# TODO: Can you somehow incorporate precision and recall into the cost function for optimization?

# Typical workflow using tensorflow estimators API:
    # Loading the libraries and dataset.
    # Data proprocessing.
    # Defining the feature columns.
    # Building input function.
    # Model instantiation, training and evaluation.
    # Generating prediction.
    # Visualizing the model and the loss metrics using Tensorboard.



