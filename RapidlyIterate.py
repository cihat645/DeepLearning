import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, Normalizer, PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score, recall_score, precision_score
from sklearn.manifold import TSNE
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping
from keras import Model, regularizers
from keras.callbacks import  ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Input, Lambda
from keras import optimizers
from sklearn.model_selection import KFold
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout, InputLayer
from keras.models import load_model
from keras import backend as K

FOLDER_PATH = r'FILE PATH'                             
data = pd.read_csv(r'DATA PATH')
FEATURES = list(data)[:-1]
data = np.array(data)
M = data.shape[0]
# print(data.isnull().values.any())                                                                                     # check for na values

def one_hot(data, save_path, name):
    """
    This function does the same thing as the function above, except more succinctly using the one-hot package from sklearn.
    :param data:  original data we want to encode (raw data)
    :param save_path: where we save
    :param name: name of file
    :return:
    """
    labels = data[:, -1].reshape(data.shape[0], 1)                                                                      # separate labels from data
    data = data[:, :-1]
    nominal_cols = [1, 2, 3, 5, 6, 7, 8, 9, 10]                                                                         # columns that are categorical
    cont_cols = [x for x in range(data.shape[1]) if x not in nominal_cols]                                              # columns that are continuous
    encoder = OneHotEncoder(sparse= False)                                                                              # create encoder
    list_of_one_hot_mats = []
    continuous_vars = np.zeros(shape=(data.shape[0], data.shape[1] - len(nominal_cols)))                                # holds continuous variables, we will concat with nominal variables later

    for col, new_col in zip(cont_cols, range(len(cont_cols))):                                                          # fill continuous matrix
        continuous_vars[:, new_col] = data[:, col]

    for col in nominal_cols:
        min_val = min(data[:, col])
        if min_val < 0:                                                                                                 # one hot from sklearn can only handles positive ints
            data[:, col] += (-1 * min_val)                                                                              # if we have negative int, shift values up
        feat_mat = encoder.fit_transform(data[:, col].reshape(data.shape[0], 1))                                        # create list of one hot encoded matrices
        list_of_one_hot_mats.append(feat_mat)

    nominal_features = np.concatenate((list_of_one_hot_mats), axis=1)                                                   # concatenate all categorical variables
    new_data = np.concatenate((continuous_vars, nominal_features), axis=1)                                              # concatenate continuous vars w/ categorical
    new_data = np.concatenate((new_data, labels), axis=1)                                                               # raw data without preprocessing
    np.savetxt(save_path + name + '.csv', new_data, delimiter=',')                                                      # save file to disk


def preprocess_data(data, technique, labels = True):
    data_labels = None
    if labels:
        # data_labels = data[:, -1].reshape(data.shape[0], 1)
        data_labels = data[:, -1]
        data = data[:, :-1]
    if technique == "MinMax":
        transformed = MinMaxScaler().fit_transform(data)
    elif technique == "Standard":
        transformed = StandardScaler().fit_transform(data)
    elif technique == "Robust":
        transformed = RobustScaler().fit_transform(data)
    elif technique == "MaxAbs":
        transformed = MaxAbsScaler().fit_transform(data)
    elif technique == "Quantile":
        transformed = QuantileTransformer().fit_transform(data)
    elif technique == "Normalizer":
        transformed = Normalizer().fit_transform(data)
    elif technique == "Polynomial":
        transformed = PolynomialFeatures().fit_transform(data)
    else:
        print("Not a valid preprocessing method for this function")
        return None

    if labels:
        return transformed, data_labels
    else:
        return transformed

# ------------- DATA SETUP AND VISUALIZATION -----------------------------------
def plot_3d(data, colors, title):
    x, y, z = data.T
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color=colors)
    plt.title(title)
    # ax.scatter(x, y, z)
    plt.show()

def visualize_data(data):
    labels = data[:, -1]
    data = data[:, :-1]

    # --------- t-SNE Visualization ---------
    random_indices = np.random.permutation(data.shape[0])[0:400]     # generate a random sample to be displayed
    for preprocess in [MinMaxScaler(), StandardScaler(), RobustScaler(), MaxAbsScaler(), QuantileTransformer(), Normalizer(), PolynomialFeatures(), None]:
        new_data = preprocess.fit_transform(data) if preprocess else data # preprocess data and at the end, plot original data

        # 2D Visualization
        tsne = TSNE(n_components= 2, verbose=1, perplexity = 15, learning_rate= 80, n_iter= 500)
        results = tsne.fit_transform(new_data[random_indices, :])
        colors = np.array(['red' if label == 1 else 'green' for label in labels[random_indices]])
        plt.scatter(results[:, 0], results[:, 1], c =colors)
        plt.title('t-SNE 2D Visualization ' + str(preprocess).split('(')[0])
        plt.show()

        tsne = TSNE(n_components=3, verbose = 1, perplexity=15, learning_rate= 80, n_iter=1000)
        results_3d = tsne.fit_transform(new_data[random_indices, :])
        colors_3d = np.array(['red' if label == 1 else 'green' for label in labels])[random_indices]
        plot_3d(results_3d, colors_3d, title='t-SNE 3D Normalized Data ' + str(preprocess).split('(')[0])

    # ------- PCA --------
    for preprocess in [MinMaxScaler(), StandardScaler(), RobustScaler(), MaxAbsScaler(), QuantileTransformer(), Normalizer(), PolynomialFeatures(), None]:
        new_data = preprocess.fit_transform(data) if preprocess else data  # preprocess data and at the end, plot original data

        # 2D Visualization
        pca = PCA(n_components=2)
        pca.fit(new_data)
        results = pca.transform(new_data)
        colors = np.array(['red' if label == 1 else 'green' for label in labels])
        results, colors = results[random_indices,:], colors[random_indices]
        plt.scatter(results[:, 0], results[:, 1], c=colors)
        plt.title('PCA Visualization 2D ' + str(preprocess).split('(')[0])
        plt.xlabel('1st Principal Component')
        plt.ylabel('2nd Principal Component')
        plt.show()
        plt.close()

        # 3D Visualization
        pca_3d = PCA(n_components=3)
        pca_3d.fit(new_data)
        results_3d = pca_3d.transform(new_data)
        colors_3d = np.array(['red' if label == 1 else 'green' for label in labels])
        plot_3d(results_3d[random_indices, :], colors_3d[random_indices], 'PCA 3D Visualization ' + str(preprocess).split('(')[0])

def split_data(train_portion, val_portion, data):
    val_begin = round(0.8 * M) ; val_end = round(val_begin + M * val_portion)

    train_set = data[: val_begin, :-1]
    val_set = data[val_begin : val_end, :-1]
    test_set = data[val_end :, :-1]

    train_labels = data[: val_begin, -1]
    val_labels = data[val_begin: val_end, -1]
    test_labels = data[val_end:, -1]

    data_dict = {'Training data' : train_set, 'Validation data': val_set, "Test data" : test_set, 'Training labels' : train_labels, 'Validation labels':val_labels, "Test labels": test_labels}
    return data_dict

def plot_data_distributions(train_set, train_labels, val_set, val_labels, test_set, test_labels):
    # Visualize Proportions of positive defaults in each set:
    grid = GridSpec(2, 2)
    train_defaults = sum(train_labels == 1) ; val_defaults = sum(val_labels == 1) ; test_defaults = sum(test_labels == 1)
    plt.subplot(grid[0, 0], aspect = 1, title= 'Training Set')
    plt.pie([train_defaults, train_set.shape[0] - train_defaults], labels=['Defaults', 'Pass'], autopct='%1.1f%%')
    plt.subplot(grid[1, 0], aspect = 1, title= 'Validation Set')
    plt.pie([val_defaults, val_set.shape[0] - val_defaults], labels=['Defaults', 'Pass'], autopct='%1.1f%%')
    plt.subplot(grid[0, 1], aspect = 1, title = "Test Set")
    plt.pie([test_defaults, test_set.shape[0] - test_defaults], labels=['Defaults', 'Pass'], autopct='%1.1f%%')
    plt.subplot(grid[1, 1], aspect = 1, title = "Aggregate Data")
    plt.pie([sum(data[:, -1] == 1), m - sum(data[:, -1])],  labels=['Defaults', 'Pass'], autopct='%1.1f%%')
    plt.show()

# ---------------------------------------------------- Normalizing & Visualizing Data ----------------------------------------------------

# one_hot(data, FOLDER_PATH, 'OG_one_hot')                                                                                                    # one-hot enocde data and save. data is loaded above ^
# og_one_hot = np.array(pd.read_csv(r'PATH HERE'))
# print("og one hot shape = ", og_one_hot.shape)

# mean_normalized_data = preprocess_data(data, 'MeanNormalized', save= True, save_name= "\CC_Data_ScaledCC_Data_MeanNormalized")              # testing preprocessing functions
# mean_normalized_data = preprocess_data(data, 'Scaled', save= True, save_name= "\CC_Data_ScaledCC_Data_Scaled")

# m = data.shape[0]
# train_portion = 0.8 ; val_portion = 0.1                                                                                                       #  Set proportions of data segments
#
# visualize_data(data)
# normalized_data = np.array(pd.read_csv(FOLDER_PATH + '\Credit_Card_Scaled.csv'))
# data_dict = split_data(train_portion, val_portion, normalized_data)
#
# data_dict = split_data(train_portion, val_portion, data)
# visualize_data(normalized_data)

# ------------------------------------------------------------------------------------------------------------------------------------------

# ------------------ MODELING FUNCTIONS ---------------------------------

def build_nn(model_info):
    """
    This function builds and compiles a NN given a hash table of the model's parameters.
    :param model_info:
    :return:
    """
    K.clear_session()                                                           # force Keras TF backend to start a new session
    try:
        if model_info["Regularization"] == "l2":                                # if we're using L2 regularization
            lambda_ = model_info['Reg param']                                   # get lambda parameter
            batch_norm, keep_prob = False, False                                # set other regularization tactics

        elif model_info['Regularization'] == 'Batch norm':                      # batch normalization regularization
            lambda_ = 0
            batch_norm = model_info['Reg param']                                # get param
            keep_prob = False
            if batch_norm not in ['before', 'after']:                           # ensure we have a valid reg param
                raise ValueError

        elif model_info['Regularization'] == 'Dropout':                         # Dropout regularization
            lambda_, batch_norm = 0, False
            keep_prob = model_info['Reg param']
    except:
        lambda_, batch_norm, keep_prob = 0, False, False                        # if no regularization is being used

    hidden, acts = model_info['Hidden layers'], model_info['Activations']
    model = Sequential(name=model_info['Name'])
    model.add(InputLayer((model_info['Input size'],)))                            # create input layer
    first_hidden = True

    for lay, act, i in zip(hidden, acts, range(len(hidden))):                                          # create all the hidden layers
        if lambda_ > 0:                                                         # if we're doing L2 regularization
            if not first_hidden:
                model.add(Dense(lay, activation=act, W_regularizer=l2(lambda_), input_shape=(hidden[i - 1],)))    # add additional layers
            else:
                model.add(Dense(lay, activation=act, W_regularizer=l2(lambda_), input_shape=(model_info['Input size'],)))
                first_hidden = False
        else:                                                                   # if we're not regularizing
            if not first_hidden:
                model.add(Dense(lay, input_shape=(hidden[i-1], )))              # add un-regularized layers
            else:
                model.add(Dense(lay, input_shape=(model_info['Input size'],)))  # if its first layer, connect it to the input layer
                first_hidden = False

        if batch_norm == 'before':
            model.add(BatchNormalization(input_shape=(lay,)))               # add batch normalization layer

        model.add(Activation(act))                                          # activation layer is part of the hidden layer

        if batch_norm == 'after':
            model.add(BatchNormalization(input_shape=(lay,)))               # add batch normalization layer

        if keep_prob:
            model.add(Dropout(keep_prob, input_shape=(lay,)))               # dropout layer

    # --------- Adding Output Layer -------------
    model.add(Dense(1, input_shape=(hidden[-1], )))                             # add output layer
    if batch_norm == 'before':                                                  # if we're using batch norm regularization
        model.add(BatchNormalization(input_shape=(hidden[-1],)))
    model.add(Activation('sigmoid'))                                            # apply output layer activation
    if batch_norm == 'after':
        model.add(BatchNormalization(input_shape=(hidden[-1],)))                # adding batch norm layer

    if model_info['Optimization'] == 'adagrad':                                 # setting an optimization method
        opt = optimizers.Adagrad(lr = model_info["Learning rate"])
    elif model_info['Optimization'] == 'rmsprop':
        opt = optimizers.RMSprop(lr = model_info["Learning rate"])
    elif model_info['Optimization'] == 'adadelta':
        opt = optimizers.Adadelta()
    elif model_info['Optimization'] == 'adamax':
        opt = optimizers.Adamax(lr = model_info["Learning rate"])
    else:
        opt = optimizers.Nadam(lr = model_info["Learning rate"])
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])  # compile model

    return model


def k_fold_cross_validation(param_dict, data, preproc_tech, model=None, num_epochs = 15):

    kf = KFold(n_splits=5, shuffle=True)
    results = []                                                    # list of the model's training accuracy, test accuracy, train AUC, and test AUC for each fold of the k-fold cross validation

    for train_indices, test_indices in kf.split(data[:, :-1]):
        train_data, train_labels = preprocess_data(data[train_indices, :], preproc_tech, True)                  # preprocess data in the CV loop to circumvent evaluation bias
        test_data, test_labels = preprocess_data(data[test_indices, :], preproc_tech, True)

        if not model:                                                                                           # if we don't provide a model, build one
            model = build_nn(param_dict)
        model.fit(train_data, train_labels, epochs=num_epochs, batch_size=param_dict["Batch size"], verbose=0)  # train model
        y_pred = model.predict(train_data).ravel()                                                              # predict on training data
        fpr, tpr, thresholds = roc_curve(train_labels, y_pred)                                                  # compute fpr and tpr
        auc_train = auc(fpr, tpr)                                                                               # compute AUC metric
        _, train_acc = model.evaluate(train_data, train_labels, verbose=0)                                      # evaluate on training data

        y_pred = model.predict(test_data).ravel()                                                               # same as above with test data
        fpr, tpr, thresholds = roc_curve(test_labels, y_pred)                                                   # compute FPR & TPR
        auc_test = auc(fpr, tpr)                                                                                # Get AUC for test set
        _, test_acc = model.evaluate(test_data, test_labels, verbose=0)                                         # evaluate model
        results.append((train_acc, test_acc, auc_train, auc_test))                                              # append results

    return results

def display_k_fold_results(model_hist, top_models = 1):
    """
    Displays results of k-fold cross validation test.
    :param model_hist:
    :param top_models: the number of top models to display (e.g. if you're testing 40 models, this will only show the top_models best performing models
    :return:
    """

    print('------------------ RESULTS FOR ' + model_hist[0]['Preprocessing'] + " PREPROCESSING ------------------------------------")
    top_test_auc = sorted(model_hist, key=lambda k: k['Avg Test AUC'])
    print('\nTOP Avg AUC Test:\n')
    for a_model in top_test_auc[-top_models:]:
        print('-------------------------\nLearning Rate: ', a_model['Learning rate'], '\nBatch size: ', a_model['Batch size'])
        print('Avg Train AUC: ', a_model["Avg Train AUC"], '\nAvg Test AUC: ', a_model["Avg Test AUC"], '\nAvg Train Accuracy: ', a_model["Avg Train Accuracy"], "\nAvg Test Accuracy", a_model["Avg Test Accuracy"])

    top_train_auc = sorted(model_hist, key=lambda k: k['Avg Train AUC'])
    print('\nTOP Avg AUC Train:\n\n')
    for a_model in top_train_auc[-top_models:]:
        print('-------------------------\nLearning Rate: ', a_model['Learning rate'], '\nBatch size: ', a_model['Batch size'])
        print('Avg Train AUC: ', a_model["Avg Train AUC"], '\nAvg Test AUC: ', a_model["Avg Test AUC"], '\nAvg Train Accuracy: ', a_model["Avg Train Accuracy"], "\nAvg Test Accuracy", a_model["Avg Test Accuracy"])

    top_test_acc = sorted(model_hist, key=lambda k: k["Avg Test Accuracy"])
    print('\nTOP Avg Test Accuracy:\n\n')
    for a_model in top_test_acc[-top_models:]:
        print('-------------------------\nLearning Rate: ', a_model['Learning rate'], '\nBatch size: ', a_model['Batch size'])
        print('Avg Train AUC: ', a_model["Avg Train AUC"], '\nAvg Test AUC: ', a_model["Avg Test AUC"], '\nAvg Train Accuracy: ', a_model["Avg Train Accuracy"], "\nAvg Test Accuracy", a_model["Avg Test Accuracy"])

    top_train_acc = sorted(model_hist, key=lambda k: k["Avg Train Accuracy"])
    print('\nTOP Avg Train Accuracy:\n\n')
    for a_model in top_train_acc[-top_models:]:
        print('-------------------------\nLearning Rate: ', a_model['Learning rate'], '\nBatch size: ', a_model['Batch size'])
        print('Avg Train AUC: ', a_model["Avg Train AUC"], '\nAvg Test AUC: ', a_model["Avg Test AUC"], '\nAvg Train Accuracy: ', a_model["Avg Train Accuracy"], "\nAvg Test Accuracy", a_model["Avg Test Accuracy"])
    print(
        "------------------------------------------------------------------------------------------------------------")

def display_model_info(model_info):
    print('\n---------------------------------------------------------,')
    print("Architecture:\nLayers: ", [model_info['Input size']] + [model_info['Hidden layers']] + [1])                                                                   # show layers & number of units per layer
    print("Activations: ", model_info["Activations"] + ["sigmoid"])                                                                                                      # print hidden layer activations
    print("Hyperparameters:\nBatch size: ", model_info["Batch size"], "\nLearning rate: ", model_info["Learning rate"], "\nOptimization: ", model_info["Optimization"])
    if model_info["Preprocessing"]:
        print("Preprocessing: ", model_info["Preprocessing"])
    print("K-Fold CV RESULTS:\nAvg Train AUC: ", model_info["Avg Train AUC"], "\nAvg Test AUC: ", model_info["Avg Test AUC"], "\nAvg Train Accuracy: ", model_info["Avg Train Accuracy"])
    print("Avg Test Accuracy: ", model_info["Avg Test Accuracy"])

def generate_random_model():
    optimization_methods = ['adagrad', 'rmsprop', 'adadelta', 'adam', 'adamax', 'nadam']      # possible optimization methods
    activation_functions = ['sigmoid', 'relu', 'tanh']          # possible activation functions
    batch_sizes = [16, 32, 64, 128, 256, 512]                   # possible batch sizes
    range_hidden_units = range(5, 250)                          # range of possible hidden units
    model_info = {}                                             # create hash table
    same_units = np.random.choice([0, 1], p=[1/5, 4/5])         # dictates whether all hidden layers will have the same number of units
    same_act_fun = np.random.choice([0, 1], p=[1/10, 9/10])     # will each hidden layer have the same activation function?
    really_deep = np.random.rand()
    range_layers = range(1, 10) if really_deep < 0.8 else range(6, 20)          # 80% of time constrain number of hidden layers between 1 - 10, 20% of time permit really deep architectures
    num_layers = np.random.choice(range_layers, p=[.1, .2, .2, .2, .05, .05, .05, .1, .05]) if really_deep < 0.8 else np.random.choice(range_layers)    # choose number of layers
    model_info["Activations"] = [np.random.choice(activation_functions, p = [0.25, 0.5, 0.25])] * num_layers if same_act_fun else [np.random.choice(activation_functions, p = [0.25, 0.5, 0.25]) for _ in range(num_layers)] # choose activation functions
    model_info["Hidden layers"] = [np.random.choice(range_hidden_units)] * num_layers if same_units else [np.random.choice(range_hidden_units) for _ in range(num_layers)]  # create hidden layers
    model_info["Optimization"] = np.random.choice(optimization_methods)         # choose an optimization method at random
    model_info["Batch size"] = np.random.choice(batch_sizes)                    # choose batch size
    model_info["Learning rate"] = 10 ** (-4 * np.random.rand())                 # choose a learning rate on a logarithmic scale
    model_info["Training threshold"] = 0.5                                      # set threshold for training
    return model_info

def quick_nn_test(model_info, data_dict, save_path):
    model = build_nn(model_info)                                    # use model info to build and compile a nn
    stop = EarlyStopping(patience=5, monitor='acc', verbose=1)      # maintain a max accuracy for a sliding window of 5 epochs. If we cannot breach max accuracy after 15 epochs, cut model off and move on.
    tensorboard_path = save_path + model_info['Name'] + '\\'        # create path for tensorboard callback
    tensorboard = TensorBoard(log_dir=tensorboard_path, histogram_freq=0, write_graph=True, write_images=True)              # create tensorboard callback
    save_model = ModelCheckpoint(filepath= save_path + model_info['Name'] + '\\' + model_info['Name'] + '_saved_' + '.h5')  # save model after every epoch

    model.fit(data_dict['Training data'], data_dict['Training labels'], epochs=3,                 # fit model
              batch_size=model_info['Batch size'], callbacks=[save_model, stop, tensorboard])     # evaluate train accuracy

    train_acc = model.evaluate(data_dict['Training data'], data_dict['Training labels'],
                               batch_size=model_info['Batch size'], verbose=0)
    test_acc = model.evaluate(data_dict['Test data'], data_dict['Test labels'],                     # evaluate test accuracy
                              batch_size=model_info['Batch size'], verbose=0)

                                                                                            # Get Train AUC
    y_pred = model.predict(data_dict['Training data']).ravel()                          # predict on training data
    fpr, tpr, thresholds = roc_curve(data_dict['Training labels'], y_pred)              # compute fpr and tpr
    auc_train = auc(fpr, tpr)                                                           # compute AUC metric
                                                                                        # Get Test AUC
    y_pred = model.predict(data_dict['Test data']).ravel()                              # same as above with test data
    fpr, tpr, thresholds = roc_curve(data_dict['Test labels'], y_pred)                  # compute AUC
    auc_test = auc(fpr, tpr)

    return train_acc, test_acc, auc_train, auc_test
    # return 9, 9, 9, 9

def test_nn_models(num_models, raw_data, preprocess_tech, save_path, eval_tech = 'k-fold', list_of_models= None):
    """
    This function is used to evaluate neural network performance on a dataset. We can feed it either a list of models to test explicitly, or if we leave this parameter as None,
    it will randomly generate neural networks.
    :param num_models:      number of models we want to test
    :param raw_data:        raw data without any preprocessing done
    :param preprocess_tech: the technique we want to employ in preprocessing the data
    :param save_path:       where to save the results df
    :param eval_tech:       how we want to evaluate the performance of the neural network.
    :param list_of_models: if provided, it will test every neural network in the list. Models must be pre-compiled.
    :return:
    """
    model_hist = [] ; model_results = pd.DataFrame(columns=['Hidden', 'Activations', 'Learning rate', 'Train Accuracy', 'Test Accuracy', 'Train AUC', 'Test AUC'])  # TODO: Add tensorboard and model saving capabilities to this function
    result_index = 0
    if eval_tech not in ['k-fold', 'boot']:                                                                 # must be a valid evaluation technique
        raise NameError

    for i in range(num_models):
        if not list_of_models:                                                                              # if a list of models to test isn't provided
            model_info = generate_random_model()                                                            # randomly generate models
            model_info["Preprocessing"] = preprocess_tech
            model_info["Input size"] = raw_data.shape[1] - 1                                                # set input data size
        else:
            assert num_models == len(list_of_models)                                                        # if we provide models, ensure the num_models param lines up
            model_info = list_of_models[i]                                                                  # get the ith model's info

        model_info["Model type"] = 'NN'                                                                     # this is for k-fold cross validation, since k-fold is able to evaluate both neural nets and logistic regression models
        model = build_nn(model_info)                                                                        # build & compile NN

        if eval_tech == 'k-fold':
            results = k_fold_cross_validation(model_info, raw_data, preprocess_tech, model=model)           # do k-fold cross validation on model
        else:
            print("currently not supporting bootstrap evaluation due to computational constraints")

        train_acc, test_acc, train_auc, test_auc = zip(*results)                                            # separate results into lists

        model_info["Avg Test AUC"] = round(np.mean(test_auc), 3)                                            # compute average metrics for each fold of the evaluation
        model_info["Avg Train AUC"] = round(np.mean(train_auc), 3)
        model_info["Avg Train Accuracy"] = round(np.mean(train_acc), 3)                                     # these are averages from k-fold cross validation
        model_info["Avg Test Accuracy"] = round(np.mean(test_acc), 3)
        model_hist.append(model_info)                                                                       # add this model's info to the list

        model_results.loc[result_index] = (model_info['Hidden layers'], model_info['Activations'],\
                                           model_info['Learning rate'], model_info['Avg Train Accuracy'],\
                                           model_info["Avg Test Accuracy"], \
                                           model_info["Avg Train AUC"], model_info["Avg Test AUC"])         # write model results to df
        result_index += 1

    model_results.to_csv(save_path + "K-fold NN Results.csv")                                               # save results to disk
    display_k_fold_results(model_hist, top_models=1)                                                        # display the top models' performance


def create_five_nns(input_size, hidden_size, act=None):
    """
    Creates 5 neural networks to be used as a baseline in determining the influence model depth & width has on performance.
    :param input_size:
    :param hidden_size:
    :param act: activation function to use for each layer
    :return:
    """
    act = ['relu'] if not act else [act]                             # default activation = 'relu'
    nns = []                                                         # list of model info hash tables
    model_info = {}                                                  # hash tables storing model information
    model_info['Hidden layers'] = [hidden_size]
    model_info['Input size'] = input_size
    model_info['Activations'] = act
    model_info['Optimization'] = 'adadelta'
    model_info["Learning rate"] = .005
    model_info["Batch size"] = 32
    model_info["Preprocessing"] = 'Standard'
    model_info2, model_info3, model_info4, model_info5 = model_info.copy(), model_info.copy(), model_info.copy(), model_info.copy()

    model_info["Name"] = 'Shallow NN'                                 # build shallow nn
    nns.append(model_info)

    model_info2['Hidden layers'] = [hidden_size] * 3                  # build medium nn
    model_info2['Activations'] = act * 3
    model_info2["Name"] = 'Medium NN'
    nns.append(model_info2)

    model_info3['Hidden layers'] = [hidden_size] * 6                  # build deep nn
    model_info3['Activations'] = act * 6
    model_info3["Name"] = 'Deep NN 1'
    nns.append(model_info3)

    model_info4['Hidden layers'] = [hidden_size] * 11                 # build really deep nn
    model_info4['Activations'] = act * 11
    model_info4["Name"] = 'Deep NN 2'
    nns.append(model_info4)

    model_info5['Hidden layers'] = [hidden_size] * 20                   # build realllllly deep nn
    model_info5['Activations'] = act * 20
    model_info5["Name"] = 'Deep NN 3'
    nns.append(model_info5)
    return nns

# ------------------- ESTABLISH BASELINES w/ 5-FOLD CROSS VALIDATION -------------------
# Test Logistic regression on original data preprocessed and og_one_hot preprocessed
og_one_hot = np.array(pd.read_csv(r'C:\Users\x2016onq\PycharmProjects\CreditDefault\OG_one_hot.csv'))

# print("TESTING.....")

# Evaluate several models via 5-Fold Cross validation
save_path = r'YOUR PATH HERE'

# path = save_path + r'\50 sigmoid\\'                                                             # test nns w/ 50 neurons on one-hot
# list_of_nns = create_five_nns(input_size=og_one_hot.shape[1]-1, hidden_size=50, act='sigmoid')
# test_nn_models(len(list_of_nns), og_one_hot, 'Standard', path, list_of_models=list_of_nns)
#
# path = save_path + r'\50 tanh\\'                                                                # test nns w/ 50 neurons on one-hot
# list_of_nns = create_five_nns(input_size=og_one_hot.shape[1]-1, hidden_size=50, act='tanh')
# test_nn_models(len(list_of_nns), og_one_hot, 'Standard', path, list_of_models=list_of_nns)
#
# path = save_path + r'\40\\'                                                                     # test nns w/ 40 neurons on one-hot
# list_of_nns = create_five_nns(input_size=og_one_hot.shape[1]-1, hidden_size=40)
# test_nn_models(len(list_of_nns), og_one_hot, 'Standard', path, list_of_models=list_of_nns)
#
# path = save_path + r'\60\\'                                                                     # test nns w/ 40 neurons on one-hot
# list_of_nns = create_five_nns(input_size=og_one_hot.shape[1]-1, hidden_size=60)
# test_nn_models(len(list_of_nns), og_one_hot, 'Standard', path, list_of_models=list_of_nns)
#
# path = save_path + r'\70\\'                                                                     # test nns w/ 40 neurons on one-hot
# list_of_nns = create_five_nns(input_size=og_one_hot.shape[1]-1, hidden_size=70)
# test_nn_models(len(list_of_nns), og_one_hot, 'Standard', path, list_of_models=list_of_nns)
#
# path = save_path + r'\80\\'                                                                     # test nns w/ 40 neurons on one-hot
# list_of_nns = create_five_nns(input_size=og_one_hot.shape[1]-1, hidden_size=80)
# test_nn_models(len(list_of_nns), og_one_hot, 'Standard', path, list_of_models=list_of_nns)
#
# path = save_path + r'\90\\'                                                                     # test nns w/ 40 neurons on one-hot
# list_of_nns = create_five_nns(input_size=og_one_hot.shape[1]-1, hidden_size=90)
# test_nn_models(len(list_of_nns), og_one_hot, 'Standard', path, list_of_models=list_of_nns)
#
# path = save_path + r'\100\\'                                                                    # test nns w/ 40 neurons on one-hot
# list_of_nns = create_five_nns(input_size=og_one_hot.shape[1]-1, hidden_size=100)
# test_nn_models(len(list_of_nns), og_one_hot, 'Standard', path, list_of_models=list_of_nns)
#
# path = save_path + r'\110\\'                                                                    # test nns w/ 40 neurons on one-hot
# list_of_nns = create_five_nns(input_size=og_one_hot.shape[1]-1, hidden_size=110)
# test_nn_models(len(list_of_nns), og_one_hot, 'Standard', path, list_of_models=list_of_nns)
#
# path = save_path + r'\120\\'                                                                    # test nns w/ 40 neurons on one-hot
# list_of_nns = create_five_nns(input_size=og_one_hot.shape[1]-1, hidden_size=120)
# test_nn_models(len(list_of_nns), og_one_hot, 'Standard', path, list_of_models=list_of_nns)


# ------------------------ QUICK TESTING ------------------------

"""This section of code allows us to create and test many neural networks and save the results of a quick 
test into a CSV file. Once that CSV file has been created, we will continue to add results onto the existing 
file."""

rapid_testing_path = 'YOUR PATH HERE'  # TODO: UNCOMMENT THIS
data_path = 'YOUR DATA PATH' # TODO: UNCOMMENT THIS

rapid_testing_path = r'C:\Users\x2016onq\PycharmProjects\CreditDefault\Test\\'
data_path = r'C:\Users\x2016onq\PycharmProjects\CreditDefault\OG_one_hot.csv'

try:                                                                        # try to load existing csv
    rapid_mlp_results = pd.read_csv(rapid_testing_path + 'Results.csv')
    index = rapid_mlp_results.shape[1]
except:                                                                     # if no csv exists yet, create a DF
    rapid_mlp_results = pd.DataFrame(columns=['Model', 'Train Accuracy', 'Test Accuracy', 'Train AUC', 'Test AUC',
                                              'Preprocessing', 'Batch size', 'Learn Rate', 'Optimization', 'Activations',
                                              'Hidden layers', 'Regularization'])
    index = 0

og_one_hot = np.array(pd.read_csv(data_path))                     # load one hot data

model_info = {}                                                     # create model_info dicts for all the models we want to test
model_info['Hidden layers'] = [100] * 6                             # specifies the number of hidden units per layer
model_info['Input size'] = og_one_hot.shape[1] - 1                  # input data size
model_info['Activations'] = ['relu'] * 6                            # activation function for each layer
model_info['Optimization'] = 'adadelta'                             # optimization method
model_info["Learning rate"] = .005                                  # learning rate for optimization method
model_info["Batch size"] = 32
model_info["Preprocessing"] = 'Standard'                            # specifies the preprocessing method to be used

model_0 = model_info.copy()                                         # create model 0
model_0['Name'] = 'Model0'

model_1 = model_info.copy()                                         # create model 1
model_1['Hidden layers'] = [110] * 3
model_1['Name'] = 'Model1'

model_2 = model_info.copy()                                         # try best model so far with several regularization parameter values
model_2['Hidden layers'] = [110] * 6
model_2['Name'] = 'Model2'
model_2['Regularization'] = 'l2'
model_2['Reg param'] = 0.0005

model_3 = model_info.copy()
model_3['Hidden layers'] = [110] * 6
model_3['Name'] = 'Model3'
model_3['Regularization'] = 'l2'
model_3['Reg param'] = 0.05

model_4 = model_info.copy()                                                             # try best model so far with several regularization parameter values
model_4['Hidden layers'] = [110] * 6
model_4['Name'] = 'Model4'
model_4['Regularization'] = 'l2'
model_4['Reg param'] = 0.0005

# .... create more models ....

#-------------- REGULARIZATION OPTIONS -------------
#   L2 Regularization:      Regularization: 'l2',           Reg param: lambda value
#   Dropout:                Regularization: 'Dropout',      Reg param: keep_prob
#   Batch normalization:    Regularization: 'Batch norm',   Reg param: 'before' or 'after'

models = [model_0, model_1, model_2]                                  # make a list of model_info hash tables

column_list = ['Model', 'Train Accuracy', 'Test Accuracy', 'Train AUC', 'Test AUC', 'Preprocessing',
               'Batch size', 'Learn Rate', 'Optimization', 'Activations', 'Hidden layers',
               'Regularization', 'Reg Param']

for model in models:                                                                                          # for each model_info in list of models to test, test model and record results
    train_data, labels = preprocess_data(og_one_hot, model['Preprocessing'], True)                            # preprocess raw data
    data_dict = split_data(0.9, 0, np.concatenate((train_data, labels.reshape(29999, 1)), axis=1))             # split data
    train_acc, test_acc, auc_train, auc_test = quick_nn_test(model, data_dict, save_path=rapid_testing_path)  # quickly assess model

    try:
        reg = model['Regularization']                                             # set regularization parameters if given
        reg_param = model['Reg param']
    except:
        reg = "None"                                                              # else set NULL params
        reg_param = 'NA'

    val_lis = [model['Name'], train_acc[1], test_acc[1], auc_train, auc_test, model['Preprocessing'],
                model["Batch size"], model["Learning rate"], model["Optimization"], str(model["Activations"]),
                str(model["Hidden layers"]), reg, reg_param]

    df_dict = {}
    for col, val in zip(column_list, val_lis):                                    # create df dict to append to csv file
        df_dict[col] = val

    df = pd.DataFrame(df_dict, index=[index])
    rapid_mlp_results = rapid_mlp_results.append(df, ignore_index=False)
    rapid_mlp_results.to_csv(rapid_testing_path + "Results.csv", index=False)


