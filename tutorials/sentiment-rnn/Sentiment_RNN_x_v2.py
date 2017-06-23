# <makedowncell>
# 2017-6-13
# tune some parameters and achieve Test Accuracy 85%
#
# <makedowncell>
# # Sentiment Analysis with an RNN
# 
# In this notebook, you'll implement a recurrent neural network that performs sentiment analysis. Using an RNN rather than a feedfoward network is more accurate since we can include information about the *sequence* of words. Here we'll use a dataset of movie reviews, accompanied by labels.
# 
# The architecture for this network is shown below.
# 
# <img src="assets/network_diagram.png" width=400px>
# 
# Here, we'll pass in words to an embedding layer. We need an embedding layer because we have tens of thousands of words, so we'll need a more efficient representation for our input data than one-hot encoded vectors. You should have seen this before from the word2vec lesson. You can actually train up an embedding with word2vec and use it here. But it's good enough to just have an embedding layer and let the network learn the embedding table on it's own.
# 
# From the embedding layer, the new representations will be passed to LSTM cells. These will add recurrent connections to the network so we can include information about the sequence of words in the data. Finally, the LSTM cells will go to a sigmoid output layer here. We're using the sigmoid because we're trying to predict if this text has positive or negative sentiment. The output layer will just be a single unit then, with a sigmoid activation function.
# 
# We don't care about the sigmoid outputs except for the very last one, we can ignore the rest. We'll calculate the cost from the output of the last step and the training label.

# <codecell>

import numpy as np
import tensorflow as tf

# <codecell>

with open('../sentiment-network/reviews.txt', 'r') as f:
    reviews = f.read()
with open('../sentiment-network/labels.txt', 'r') as f:
    labels = f.read()

reviews[:300]


# ## Data preprocessing
# 
# The first step when building a neural network model is getting your data into the proper form to feed into the network. Since we're using embedding layers, we'll need to encode each word with an integer. We'll also want to clean it up a bit.
# 
# You can see an example of the reviews data above. We'll want to get rid of those periods. Also, you might notice that the reviews are delimited with newlines `\n`. To deal with those, I'm going to split the text into each review using `\n` as the delimiter. Then I can combined all the reviews back together into one big string.
# 
# First, let's remove all punctuation. Then get all the text without the newlines and split it into individual words.

# <codecell>

from string import punctuation
all_text = ''.join([c for c in reviews if c not in punctuation])
reviews = all_text.split('\n')

# list of string, one item one review
print(reviews[:3])

# one big string of all words
all_text = ' '.join(reviews)
words = all_text.split()

# <codecell>

print(all_text[:2000])

# list of all words, one item one word. len=6020196
print(words[:100])
print(len(words))

# <codecell>

# Create your dictionary that maps vocab words to integers here
# word set, len=74072
vocab_set = set(words)
print(len(vocab_set))

# dict, vocab to int, int value starts at 1, leave 0 for padding
vocab_to_int = {word : i+1 for i, word in enumerate(vocab_set)}

# Convert the reviews to integers, same shape as reviews list, but with integers
# review list, one item one review, but int represent word
reviews_ints = []
for review in reviews:
    words_in_one_review = review.split()
    one_review_int = []
    for word in words_in_one_review:
        word_int = vocab_to_int[word]
        one_review_int.append(word_int)
    reviews_ints.append(one_review_int)
    
print(reviews_ints[0:2])
print(len(reviews_ints))

# <codecell>

# split labels
labels = labels.split('\n')
len(labels)

# <codecell>

from collections import Counter
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))

# <codecell>

# Filter out that review with 0 length
for i, e in enumerate(reviews_ints):
    if(len(e) == 0):
        print(i)
        reviews_ints.remove(e)
        del labels[i]
        break

# make sure reviews and labels still the same size
print(len(reviews_ints))
print(len(labels))

# <codecell>

# truncate review to the limitation "seq_len"
# left pad "0" to reviews whose words less than "seq_len"

seq_len = 500
# np array 2d, shape=(num_reviews, seq_len), the reviews are truncated to seq_len
features = np.zeros(shape=(len(reviews_ints), seq_len))
for i, e in enumerate(reviews_ints):
#    e = np.asarray(e)
#    print(i, e[0:3])
    if(len(e) >= seq_len):
        features[i,:] = e[0:seq_len]
    else:
        features[i,seq_len-len(e):] = e
        
print(features[:10,:100])
print(features.shape)

# <codecell>

# construct labels array with 0,1
dict_label = {'positive':1, 'negative':0}
labels_int = np.asarray([dict_label[word] for word in labels])
labels_int[:10]

# <codecell>

# split data set into train, valid, test set

split_frac = 0.8

train_end_index = int(len(features) * split_frac)
train_x, val_x = features[: train_end_index, :], features[train_end_index :, :]
train_y, val_y = labels_int[: train_end_index], labels_int[train_end_index :]

valid_end_index = int(len(features) * split_frac) + int(len(val_x) * 0.5)
val_x, test_x = features[train_end_index : valid_end_index, :], features[valid_end_index :, :]
val_y, test_y = labels_int[train_end_index : valid_end_index], labels_int[valid_end_index :]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))
print("\t\t\tlabels")
print("Train set: \t\t{}".format(train_y.shape), 
      "\nValidation set: \t{}".format(val_y.shape),
      "\nTest set: \t\t{}".format(test_y.shape))

# <codecell>

# parameters
lstm_size = 100
lstm_layers = 1
dropout_rate = 1.0
batch_size = 500
#learning_rate = 0.0005
epochs = 4

# vocab size
n_words = len(vocab_to_int)
# Size of the embedding vectors (number of units in the embedding layer)
embed_size = 32

# <codecell>

from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout

# model
model = Sequential()

# Embedding layer
# input_dim: vocab_size + 1
# output_dim: the embed_size of output
# input_length: input vector length
model.add(Embedding(input_dim=n_words+1, output_dim=embed_size, input_length=seq_len))

# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than input_dim.
# now model.output_shape == (None, input_length, output_dim), where None is the batch dimension.

# <codecell>

## unit test, test embedding layer --------------------
## (batch, input_length)
#batch_size = 32
#input_array = np.random.randint(n_words, size=(batch_size, seq_len))
#
#model.compile('rmsprop', 'mse')
#output_array = model.predict(input_array)
#print(output_array.shape)
#assert output_array.shape == (batch_size, seq_len, embed_size)
## unit test end --------------------

# <codecell>

#model.add(LSTM(
#    input_dim=embed_size,
#    output_dim=lstm_size,
#    return_sequences=True))
model.add(LSTM(
    input_dim=embed_size,
    output_dim=lstm_size,
    return_sequences=False))
        
#model.add(LSTM(
#    lstm_size,
#    return_sequences=False))
model.add(Dropout(dropout_rate))

#model.add(Dense(20))
model.add(Dense(output_dim=1, activation='sigmoid'))

#model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# <codecell>
# training
history = model.fit(
    train_x,
    train_y,
    batch_size=batch_size,
    nb_epoch=epochs,
    validation_data=(val_x, val_y))


# <codecell>
# summarize loss and accuracy

# data in history
print(history.history.keys())

import matplotlib.pyplot as plt
# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

# summarize history for accuracy
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')

# <codecell>

# evaluate on test data
#pred_test = model.predict(test_x)
scores = model.evaluate(test_x, test_y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

