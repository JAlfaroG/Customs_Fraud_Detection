# Prepare environments
import os
import string
# import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split

import random
from datetime import datetime

pd.set_option('display.max_columns', 500)

plt.style.use('fivethirtyeight')

start = datetime.now()

exp_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
print(f"Experiment ID: {exp_id}")

print("Loading data...")
# Load data
df_raw = pd.read_csv('../data/mgr3-1019.csv', low_memory=False)

# only imports
df_raw = df_raw[df_raw['destinacion_mercancia'] == 'IM']

df = df_raw.copy()

del df_raw

columns_to_use = [
    'anno', 'mes', 'aduana',
    #'agente_aduanas',
    'contribuyente', 'inciso_arancelario', 'pais_origen_destino',
    'cif', 'cuantia', 'peso_bruto', 'impuesto_total', 'illicit'
]

df = df[columns_to_use]

# df = df.sample(frac=0.75)

print("Preprocessing...")
# Set 2013 data as training data
#df_train = df[(df.anno >= 2013) & (df.anno <= 2015)].reset_index(drop=True)
# Set 2014 data as testing data
#df_test = df[(df.anno >= 2016) & (df.anno <= 2016)].reset_index(drop=True)

df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)

print("Now, train-data has {} entries and {} features".format(*df_train.shape))
print("Now, test-data has {} entries and {} features".format(*df_test.shape))

# Extracting labels
train_labels = df_train.pop('illicit')
test_labels = df_test.pop('illicit')

# Saving labels
if not os.path.exists('./labels'):
    os.makedirs('./labels')

save_dir = './labels'

np.save(save_dir + '/test_labels', test_labels)
np.save(save_dir + '/train_labels', train_labels)

# Define numeric variables
numeric_vars = ['cif', 'cuantia', 'peso_bruto', 'impuesto_total']
# Define categorical variables
categorical_vars = list(set(df_train.columns)-set(numeric_vars))

# replace na with 0 or 'unknown'
df_train[categorical_vars] = df_train[categorical_vars].fillna('unknown')
df_train[numeric_vars] = df_train[numeric_vars].fillna(0)
df_test[categorical_vars] = df_test[categorical_vars].fillna('unknown')
df_test[numeric_vars] = df_test[numeric_vars].fillna(0)

# Convert categorical variables to numeric variables (option 1)

# Create an empty dict for mapping classes with labels
Categories_mapping = {}

# categorical_vars = ['inciso_arancelario', 'contribuyente', 'pais_origen_destino',]
for var in categorical_vars:
    # Option 1
    # Make sure that it is a categorical variable
    df_train[var] = df_train[var].astype(str)  # str
    # Create a mapping-reference table for mapping classes(categories) with labels
    Categories_mapping[var] = {key: value for value, key in enumerate(df_train[var].unique())}
    # Convert classes(categories) to a corresponding numeric code
    df_train[var] = df_train[var].map(Categories_mapping[var])

    '''
    # Option 2
    # Make sure that it is a categorical variable
    df_train.loc[:, var] = df_train[var].astype('category')
    # Create a mapping-reference table for mapping classes(categories) with labels
    #Categories_mapping[var] = dict(zip(df_train[var].cat.categorical, df_train[var].cat.codes))
    Categories_mapping[var] = dict(zip(pd.Categorical(df_train[var]), pd.Categorical(df_train[var].cat.codes)))
    # Convert classes(categories) to a corresponding numeric code
    df_train[var] = df_train.copy()[var].cat.codes
    '''

print("Preparing categorical embedding size...")
# https://towardsdatascience.com/categorical-embedding-and-transfer-learning-dd3c4af6345d
# https://forums.fast.ai/t/embedding-layer-size-rule/50691/13
# https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
# Set size of embedding

# Prepare an empty dictionary for saving category size (number of classes)
category_sizes = {}
# Prepare an empty dictionary for saving the embedding size (number of embedding columns)
categorical_vars_embsizes = {}

for var in categorical_vars:
    # Count the unique values in each categorical variable
    category_sizes[var] = df_train[var].nunique()
    # Set the size of embedding
    categorical_vars_embsizes[var] = min(1000, category_sizes[var] + 1//2) # min(600, round(1.6 * category_sizes[var] ** .56))

# label-encode test-data in reference to train-data
for vars in categorical_vars:
    df_test[vars] = df_test[vars].map(Categories_mapping[vars])
    df_test[vars][df_test[vars].isnull()] = df_train[vars].nunique()
df_test[categorical_vars].isnull().any()

# Scale numeric variables
# log-scale numeric variables|
print("Log-scaling train and test sets...")
df_train[numeric_vars] = df_train[numeric_vars].apply(np.log1p)
df_test[numeric_vars] = df_test[numeric_vars].apply(np.log1p)

print("Convert train data types to int32 and float32...")
for x in df_train.columns:
    if df_train[x].dtypes == 'int64':
        df_train.loc[:, x] = df_train.loc[:, x].astype(np.int32)
    elif df_train[x].dtypes == 'float64':
        df_train.loc[:, x] = df_train.loc[:, x].astype(np.float32)

print("Convert test data types to int32 and float32...")
for x in df_test.columns:
    if df_test[x].dtypes == 'int64':
        df_test.loc[:, x] = df_test.loc[:, x].astype(np.int32)
    elif df_test[x].dtypes == 'float64':
        df_test.loc[:, x] = df_test.loc[:, x].astype(np.float32)

print(f"Train data types: \n{df_train.dtypes}")
print()
print(f"Test data types: \n{df_test.dtypes}")

# print("Performing ADASYN...")
# # Oversampling
# from imblearn.over_sampling import ADASYN
#
# print("Label prop before oversampling...")
# print(train_labels.value_counts(normalize=True))
#
# ada = ADASYN()
# df_train, train_labels = ada.fit_resample(df_train, train_labels)
#
# print("Label prop after ADASYN...")
# print(train_labels.value_counts(normalize=True))

print("Converting data format to dictionary...")
X_train = {}
for vars in df_train.columns.tolist():
    X_train[vars] = df_train[vars].values
print(X_train)

X_test = {}
for vars in df_test.columns.tolist():
    X_test[vars] = df_test[vars].values
print(X_test)

# counts = train_labels.value_counts(normalize=False)
# print(
#     "Number of positive samples in training data: {} ({:.2f}% of total)".format(
#         counts[1], 100 * float(counts[1]) / len(train_labels)
#     )
# )

neg, pos = np.bincount(df['illicit'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def set_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                # tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*4)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


# set_gpu()

# Prepare deep learning
from keras.constraints import maxnorm
# from tensorflow.keras.initializers import LecunNormal
# from tensorflow.keras.layers import AlphaDropout
from keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (Dense, Dropout, Embedding, Input, Reshape, Concatenate, Flatten,
                                     BatchNormalization, LeakyReLU)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import (TruePositives, TrueNegatives, FalsePositives, FalseNegatives,
                                      Precision, Recall, AUC)

print("Preparing Deep Learning...")
# Define input layers

# Create lists to hold input layers
ins = []
concat = []

# Iterate to create input layers for categorical variables

for var in categorical_vars:
    # create an input layer for each categorical variable
    # Caution: Match the name of layer with the name of variable!!!
    # as the format of train/test data is a dictionary, where 'key()' is the variable names.
    x = Input(shape=(1,), name=var)
    # concatenate input layers for the final input layer
    ins.append(x)
    # create an embedding layer for each input layer
    x = Embedding(input_dim=category_sizes[var] + 1,
                  output_dim=categorical_vars_embsizes[var],
                  name='Embedding_' + var)(x)
    x = Flatten(name='Flatten_' + var)(x)
    # concatenate embedding layers for the final output layer
    concat.append(x)

# Iterate to create input layers for numeric variables

for var in numeric_vars:
    # create an input layer for each numeric variables
    y = Input(shape=(1,), name=var)
    # concatenate input layers for the final input layer
    ins.append(y)
    # concatenate input layers for the final output layer
    concat.append(y)

initial_bias = np.log([pos / neg])
output_bias = tf.keras.initializers.Constant(initial_bias)

# USING SELU
# https://www.machinecurve.com/index.php/2021/01/12/using-selu-with-tensorflow-and-keras/

# On Dropout
# https://www.machinecurve.com/index.php/2019/12/16/what-is-dropout-reduce-overfitting-in-your-neural-networks/

# Define output layers
output = Concatenate(name='combined')(concat)
output = Dense(1024, activation='relu', kernel_constraint=maxnorm(3))(output)
output = BatchNormalization()(output)
output = Dropout(.3)(output)
output = Dense(256, activation='relu', kernel_constraint=maxnorm(3))(output)
output = BatchNormalization()(output)
output = Dropout(.3)(output)
output = Dense(128, activation='relu', kernel_constraint=maxnorm(3))(output)
output = BatchNormalization()(output)
output = Dropout(.3)(output)
output = Dense(1, activation='sigmoid',
               activity_regularizer=l1_l2(l1=0.01, l2=0.01))(output)
#output = Dense(1, activation='sigmoid', bias_initializer=output_bias)(output)

print("Initiate model...")
# Initiate (activate) the model
model = Model(ins, output)

# Define loss function (a formula to calculate the difference between predicted value and actual value).
# Define optimizer (a rule to find the minimum value of loss function, ie. search interval, search range...)

# optimizer='sgd'
metrics = [
    FalseNegatives(name="fn"),
    FalsePositives(name="fp"),
    TrueNegatives(name="tn"),
    TruePositives(name="tp"),
    Precision(name="precision"),
    Recall(name="recall"),
    AUC(name='AUC')
]


# https://www.bigrabbitdata.com/learning-rate-scheduling-in-neural-networks/#Batch_Size_vs_Learning_Rate
# https://stackoverflow.com/questions/39517431/should-we-do-learning-rate-decay-for-adam-optimizer
# https://www.jeremyjordan.me/nn-learning-rate/
# learning_rate = 32 ** 0.5 * 0.01

# import keras.backend as K
# from keras.callbacks import Callback
from keras.callbacks import LearningRateScheduler

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''

    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))

    return LearningRateScheduler(schedule)


lr_schedule = step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=2)

# best run learning rate: 0.0001
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=metrics)

# Save weights
# import os
# if not os.path.exists('./callbacks'):
#     os.makedirs('./callbacks')

# save_dir = './callbacks'

early_stopping = EarlyStopping(
    monitor='val_AUC',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True
)

reduce_lr_auc = ReduceLROnPlateau(
    monitor='val_AUC',
    factor=0.1,
    patience=7,
    verbose=1,
    min_delta=1e-4,
    mode='max')

# model_checkpoint = ModelCheckpoint("./callbacks/fraud_model_at_epoch_{epoch}.h5", save_best_only=True, save_weights_only=True)

# callbacks = [model_checkpoint, early_stopping]

# counts = train_labels.value_counts(normalize=False)
# print(
#     "Number of positive samples in training data: {} ({:.2f}% of total)".format(
#         counts[1], 100 * float(counts[1]) / len(train_labels)
#     )
# )
#
# weight_for_0 = 1.0 / counts[0]
# weight_for_1 = 1.0 / counts[1]
# class_weight = {0: weight_for_0, 1: weight_for_1}

print("Setting up weights for balancing...")
# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg)*total/2.0
weight_for_1 = (1 / pos)*total/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

# Overview of model structure
model.summary()

print(f"Saving model {exp_id} summary...")
if not os.path.exists('./model_summaries'):
    os.makedirs('./model_summaries')

save_dir = './model_summaries'

with open(os.path.join(save_dir, f'model_{exp_id}_summary.txt'), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

print("Begin training...")

# https://stackoverflow.com/questions/53033556/how-should-the-learning-rate-change-as-the-batch-size-change
# https://stats.stackexchange.com/questions/164876/what-is-the-trade-off-between-batch-size-and-number-of-iterations-to-train-a-neu
# https://mydeeplearningnb.wordpress.com/2019/02/23/convnet-for-classification-of-cifar-10/

batch_size = 64 * 32  # 2048  # // 16
#compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / batch_size))
#steps_per_epoch = compute_steps_per_epoch(df_train.shape[0])

model.fit(
    X_train,
    train_labels,
    # steps_per_epoch=steps_per_epoch,# steps_per_epoch=int(np.ceil(df_train.shape[0] / float(batch_size))),
    epochs=3,  # 20,
    batch_size=batch_size,
    verbose=1,
    callbacks=[early_stopping, reduce_lr_auc, lr_schedule],
    validation_data=(X_test, test_labels),
    validation_split=0.1,
    class_weight=class_weight
)

# Save weights
if not os.path.exists('./embedding_results'):
    os.makedirs('./embedding_results')

save_dir = './embedding_results'
# Save autoencoder layers' weights
model.save(save_dir + f'/{exp_id}_EMB_FULL_1119.h5')

print("Saving model predictions...")
# Make predictions for test-data
y_pred = model.predict(X_test)
# transform y_pred into 1d array
y_pred = y_pred.reshape(-1)

# Make predictions for train-data
train_predictions = model.predict(X_train)
train_predictions = train_predictions.reshape(-1)

# Save the predictions of each classifier in the name of 'exp_id' as .npy files
np.save(save_dir + f'/y_pred_{exp_id}', y_pred)
np.save(save_dir + f'/y_train_{exp_id}', train_predictions)

if not os.path.exists('./embedding_plots'):
    os.makedirs('./embedding_plots')

save_dir = './embedding_plots'

fpr_train, tpr_train, _ = roc_curve(train_labels, train_predictions)

fpr, tpr, _ = roc_curve(test_labels, y_pred)

roc_auc_train = auc(fpr_train, tpr_train)
roc_auc_test = auc(fpr, tpr)
plt.figure(figsize=[10, 8])
plt.plot(fpr_train, tpr_train, color='darkorange', lw=2, label='Train AUC = %0.4f' % roc_auc_train)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Test AUC = %0.4f' % roc_auc_test, linestyle='--')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random guess')
plt.title(f'Model {exp_id} - ROC curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend(loc="lower right")
plt.savefig(save_dir + f'/{exp_id}_roc_curve.png')
plt.show()

# Evaluate by the performance of (simulated) inspection


def inspection_performance(predicted_fraud, test_fraud):
    # Set default values before a loop

    Inspect_Rate = []
    Precision = []
    Recall = []

    # Create a loop for making confusion matrix at each inspection rate

    for i in range(1, 100, 1):
        # Find the ith value in ascending order.
        threshold = np.percentile(predicted_fraud, i)
        # Precision = number of frauds / number of inspection
        precision = np.mean(test_fraud[predicted_fraud > threshold])
        # Recall = number of inspected frauds / number of frauds
        recall = sum(test_fraud[predicted_fraud > threshold]) / sum(test_fraud)
        # Save values
        Inspect_Rate.append(100 - i)
        Precision.append(precision)
        Recall.append(recall)

    compiled_conf_matrix = pd.DataFrame({

        'Inspect_Rate': Inspect_Rate,
        'Precision': Precision,
        'Recall': Recall
    })

    return compiled_conf_matrix


basic_performance = inspection_performance(y_pred, test_labels)

data = pd.melt(basic_performance,
               id_vars=['Inspect_Rate'],
               value_vars=['Recall', 'Precision'])

sns.relplot(x='Inspect_Rate',
            y='value',
            hue='variable',
            col='variable',
            kind='line',
            data=data)


# install plotly --> https://plotly.com/python/getting-started/#installation
import plotly.io
import plotly.express as px
#import plotly.offline

# Interactive visualization
fig = px.line(data, x="Inspect_Rate", y="value", color='variable', facet_col="variable")
#fig.show()

plotly.io.write_html(fig, save_dir + f"/{exp_id}_embedding_inspection_rate.html", auto_open=False)
#plotly.offline.plot(fig, filename=save_dir + "/embedding_inspection_rate.html", auto_open=False)

end = datetime.now()
print(f"Total runtime: {end - start}")