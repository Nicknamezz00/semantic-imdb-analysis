# %%
# Author: aaronwu <wrunze@foxmail.com>
print("Train CNN")
print("Author: aaronwu <wrunze@foxmail.com>")

# %%
import _pickle as cPickle
import numpy as np

from keras import backend, Input
from keras.models import Sequential
from tensorflow.python.keras.layers.core import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    Reshape,
)
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.convolutional import Convolution2D, MaxPooling2D
# WARNING:absl:At this time, the v2.11+ optimizer `tf.keras.optimizers.Adadelta` runs slowly on M1/M2 Macs,
# please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adadelta`.
# from keras.optimizers import Adadelta
from keras.optimizers.legacy import Adadelta

from keras.constraints import unit_norm
from keras.regularizers import l2
from sklearn.metrics import roc_auc_score

# %% [markdown]
# Load train, validation and test data


# %%
def get_idx_from_sent(sent, word_idx_map, max_l=51, kernel_size=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = kernel_size - 1
    for i in range(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x


def make_idx_data(revs, word_idx_map, max_l=51, kernel_size=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, val, test = [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, kernel_size)
        sent.append(rev["y"])
        if rev["split"] == 1:
            train.append(sent)
        elif rev["split"] == 0:
            val.append(sent)
        else:
            test.append(sent)
    train = np.array(train, dtype=int)
    val = np.array(val, dtype=int)
    test = np.array(test, dtype=int)
    return [train, val, test]


print("loading data...")
x = cPickle.load(open("../imdb-train-val-test.pickle", "rb"))
revs, W, word_idx_map, vocab = x[0], x[1], x[2], x[3]
print("data loaded!")

datasets = make_idx_data(revs, word_idx_map, max_l=2633, kernel_size=5)

# %% [markdown]
# Put train data in separate NumPy arrays

# %%
# Train data preparation
N = datasets[0].shape[0] # 19974
conv_input_width = W.shape[1] # 300
conv_input_height = int(datasets[0].shape[1] - 1) # 2641

# For each word write a word index (not vector) to X tensor
train_X = np.zeros((N, conv_input_height), dtype=int) # (19974, 2641)
train_Y = np.zeros((N, 2), dtype=int) # (19974, 2)
for i in range(N):
    for j in range(conv_input_height):
        train_X[i, j] = datasets[0][i, j]
    train_Y[i, datasets[0][i, -1]] = 1

print("train_X.shape = {}".format(train_X.shape))
print("train_Y.shape = {}".format(train_Y.shape))

# %% [markdown]
# Put validation data in separate NumPy arrays

# %%
# Validation data preparation
Nv = datasets[1].shape[0] # 5026

# For each word write a word index (not vector) to X tensor
val_X = np.zeros((Nv, conv_input_height), dtype=int) # (5026, 2641)
val_Y = np.zeros((Nv, 2), dtype=int) # (5026, 2)
for i in range(Nv):
    for j in range(conv_input_height):
        val_X[i, j] = datasets[1][i, j]
    val_Y[i, datasets[1][i, -1]] = 1

print("val_X.shape = {}".format(val_X.shape))
print("val_Y.shape = {}".format(val_Y.shape))
print(f'W.shape = {W.shape}')

# %% [markdown]
# Let's define and compile CNN model with Keras

# %%
backend.set_image_data_format("channels_first")

# Number of feature maps (outputs of convolutional layer)
N_fm = 300
# kernel size of convolutional layer
kernel_size = 8

model = Sequential()
model.add(Input(shape=(2641,)))
# Embedding layer (lookup table of trainable word vectors)
model.add(
    Embedding(
        input_dim=W.shape[0],
        output_dim=W.shape[1],
        input_length=conv_input_height,
        weights=[W],
    )
)
# Reshape word vectors from Embedding to tensor format suitable for Convolutional layer
model.add(Reshape((1, conv_input_height, conv_input_width)))

# first convolutional layer
model.add(
    Convolution2D(
        N_fm,
        kernel_size,
        conv_input_width,
        padding="same",
        kernel_regularizer=l2(0.0001),
    )
)
# ReLU activation
model.add(Activation("relu"))

# aggregate data in every feature map to scalar using MAX operation
model.add(MaxPooling2D(padding='same', pool_size=(conv_input_height - kernel_size + 1, 1)))

# first convolutional layer
model.add(
    Convolution2D(
        N_fm,
        kernel_size,
        conv_input_width,
        padding="same",
        kernel_regularizer=l2(0.0001),
    )
)
# ReLU activation
model.add(Activation("relu"))

# aggregate data in every feature map to scalar using MAX operation
model.add(MaxPooling2D(padding='same', pool_size=(conv_input_height - kernel_size + 1, 1)))
# first convolutional layer
model.add(
    Convolution2D(
        N_fm,
        kernel_size,
        conv_input_width,
        padding="same",
        kernel_regularizer=l2(0.0001),
    )
)
# ReLU activation
model.add(Activation("relu"))

# aggregate data in every feature map to scalar using MAX operation
model.add(MaxPooling2D(padding='same', pool_size=(conv_input_height - kernel_size + 1, 1)))

model.add(Flatten())
model.add(Dropout(0.5))
# Inner Product layer (as in regular neural network, but without non-linear activation function)
model.add(Dense(2))
# SoftMax activation; actually, Dense+SoftMax works as Multinomial Logistic Regression
model.add(Activation("softmax"))

# Custom optimizers could be used, though right now standard adadelta is employed
opt = Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-6)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()

# %%
epoch = 0
val_acc = []
val_auc = []

# %% [markdown]
# Train model for N_epoch epochs (could be run as many times as needed)

# %%
N_epoch = 5

for i in range(N_epoch):
    model.fit(train_X, train_Y, batch_size=50, epochs=1, verbose=1)
    output = model.predict(val_X, batch_size=10, verbose=1)
    # find validation accuracy using the best threshold value t
    vacc = np.max(
        [
            np.sum((output[:, 1] > t) == (val_Y[:, 1] > 0.5)) * 1.0 / len(output)
            for t in np.arange(0.0, 1.0, 0.01)
        ]
    )
    # find validation AUC
    vauc = roc_auc_score(val_Y, output)
    val_acc.append(vacc)
    val_auc.append(vauc)
    print(
        "Epoch {}: validation accuracy = {:.3%}, validation AUC = {:.3%}".format(
            epoch, vacc, vauc
        )
    )
    epoch += 1

print("{} epochs passed".format(epoch))
print("Accuracy on validation dataset:")
print(val_acc)
print("AUC on validation dataset:")
print(val_auc)

# %% [markdown]
# Save model

# %%
model.save_weights(f"cnn_{N_epoch}epochs.model")

# %% [markdown]
# Put test data in separate NumPy array

# %%
# Test data preparation
Nt = datasets[2].shape[0]

# For each word write a word index (not vector) to X tensor
test_X = np.zeros((Nt, conv_input_height), dtype=int)
for i in range(Nt):
    for j in range(conv_input_height):
        test_X[i, j] = datasets[2][i, j]

print("test_X.shape = {}".format(test_X.shape))

# %%
p = model.predict(test_X, batch_size=10)

# %% [markdown]
# Prepare submission file for Kaggle

# %%
import pandas as pd

data = pd.read_csv("../data/testData.tsv", sep="\t")
d = pd.DataFrame({"id": data["id"], "sentiment": p[:, 0]})
d.to_csv(f"cnn_{N_epoch}epochs.csv", index=False)

# %%
