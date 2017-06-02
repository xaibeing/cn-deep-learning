
# coding: utf-8

# # Handwritten Number Recognition with TFLearn and MNIST
# 
# In this notebook, we'll be building a neural network that recognizes handwritten numbers 0-9. 
# 
# This kind of neural network is used in a variety of real-world applications including: recognizing phone numbers and sorting postal mail by address. To build the network, we'll be using the **MNIST** data set, which consists of images of handwritten numbers and their correct labels 0-9.
# 
# We'll be using [TFLearn](http://tflearn.org/), a high-level library built on top of TensorFlow to build the neural network. We'll start off by importing all the modules we'll need, then load the data, and finally build the network.

# [ ]:

# Import Numpy, TensorFlow, TFLearn, and MNIST data
import numpy as np
import tensorflow as tf
import tflearn
import tflearn.datasets.mnist as mnist

import pandas as pd

# ## Retrieving training and test data
# 
# The MNIST data set already contains both training and test data. There are 55,000 data points of training data, and 10,000 points of test data.
# 
# Each MNIST data point has:
# 1. an image of a handwritten digit and 
# 2. a corresponding label (a number 0-9 that identifies the image)
# 
# We'll call the images, which will be the input to our neural network, **X** and their corresponding labels **Y**.
# 
# We're going to want our labels as *one-hot vectors*, which are vectors that holds mostly 0's and one 1. It's easiest to see this in a example. As a one-hot vector, the number 0 is represented as [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], and 4 is represented as [0, 0, 0, 0, 1, 0, 0, 0, 0, 0].
# 
# ### Flattened data
# 
# For this example, we'll be using *flattened* data or a representation of MNIST images in one dimension rather than two. So, each handwritten number image, which is 28x28 pixels, will be represented as a one dimensional array of 784 pixel values. 
# 
# Flattening the data throws away information about the 2D structure of the image, but it simplifies our data so that all of the training data can be contained in one array whose shape is [55000, 784]; the first dimension is the number of training images and the second dimension is the number of pixels in each image. This is the kind of data that is easy to analyze using a simple neural network.

# [ ]:

# Retrieve the training and test data
trainX, trainY, testX, testY = mnist.load_data(one_hot=True)


# ## Visualize the training data
# 
# Provided below is a function that will help you visualize the MNIST data. By passing in the index of a training example, the function `show_digit` will display that training image along with it's corresponding label in the title.

# [ ]:

# Visualizing the data
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Function for displaying a training image by it's index in the MNIST set
def show_digit(index):
    label = trainY[index].argmax(axis=0)
    # Reshape 784 array into 28x28 image
    image = trainX[index].reshape([28,28])
    plt.title('Training data, index: %d,  Label: %d' % (index, label))
    plt.imshow(image, cmap='gray_r')
    plt.show()
    
# Display the first (index 0) training image
show_digit(0)


# ## Building the network
# 
# TFLearn lets you build the network by defining the layers in that network. 
# 
# For this example, you'll define:
# 
# 1. The input layer, which tells the network the number of inputs it should expect for each piece of MNIST data. 
# 2. Hidden layers, which recognize patterns in data and connect the input to the output layer, and
# 3. The output layer, which defines how the network learns and outputs a label for a given image.
# 
# Let's start with the input layer; to define the input layer, you'll define the type of data that the network expects. For example,
# 
# ```
# net = tflearn.input_data([None, 100])
# ```
# 
# would create a network with 100 inputs. The number of inputs to your network needs to match the size of your data. For this example, we're using 784 element long vectors to encode our input data, so we need **784 input units**.
# 
# 
# ### Adding layers
# 
# To add new hidden layers, you use 
# 
# ```
# net = tflearn.fully_connected(net, n_units, activation='ReLU')
# ```
# 
# This adds a fully connected layer where every unit (or node) in the previous layer is connected to every unit in this layer. The first argument `net` is the network you created in the `tflearn.input_data` call, it designates the input to the hidden layer. You can set the number of units in the layer with `n_units`, and set the activation function with the `activation` keyword. You can keep adding layers to your network by repeated calling `tflearn.fully_connected(net, n_units)`. 
# 
# Then, to set how you train the network, use:
# 
# ```
# net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')
# ```
# 
# Again, this is passing in the network you've been building. The keywords: 
# 
# * `optimizer` sets the training method, here stochastic gradient descent
# * `learning_rate` is the learning rate
# * `loss` determines how the network error is calculated. In this example, with categorical cross-entropy.
# 
# Finally, you put all this together to create the model with `tflearn.DNN(net)`.

# **Exercise:** Below in the `build_model()` function, you'll put together the network using TFLearn. You get to choose how many layers to use, how many hidden units, etc.
# 
# **Hint:** The final output layer must have 10 output nodes (one for each digit 0-9). It's also recommended to use a `softmax` activation layer as your final output layer. 

# [ ]:

# Define the neural network
def build_model():
    # This resets all parameters and variables, leave this here
    tf.reset_default_graph()
    
    #### Your code ####
    # Include the input layer, hidden layer(s), and set how you want to train the model
    net = tflearn.input_data([None, 28*28])                          # Input
    net = tflearn.fully_connected(net, 30, activation='ReLU')      # Hidden
    net = tflearn.fully_connected(net, 10, activation='softmax')   # Output
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')

    # This model assumes that your network is named "net"    
    model = tflearn.DNN(net)
    return model


# [ ]:

# Build the model
model = build_model()

# 增加 callback 函数，观察 loss，acc 的变化情况
class MonitorCallback(tflearn.callbacks.Callback):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.iter = []
        self.train_losses = []
        self.train_acc = []
        self.valid_losses = []
        self.valid_acc = []

    def on_batch_end(self, training_state, snapshot=False):
#        print("The training loss is: ", training_state.global_loss)
        self.iter.append(training_state.current_iter + (training_state.epoch - 1) * self.num_samples)
        self.train_losses.append(training_state.loss_value)
        self.train_acc.append(training_state.acc_value)
        
    def on_epoch_end(self, training_state):
        self.valid_losses.append(training_state.val_loss )
        self.valid_acc.append(training_state.val_acc )
    
monitorCallback = MonitorCallback(len(trainX)*0.9)

# ## Training the network
# 
# Now that we've constructed the network, saved as the variable `model`, we can fit it to the data. Here we use the `model.fit` method. You pass in the training features `trainX` and the training targets `trainY`. Below I set `validation_set=0.1` which reserves 10% of the data set as the validation set. You can also set the batch size and number of epochs with the `batch_size` and `n_epoch` keywords, respectively. 
# 
# Too few epochs don't effectively train your network, and too many take a long time to execute. Choose wisely!

# [ ]:

# Training
# 分多次训练（运行model.fit），大约在106 epoch模型逐渐达到最优，准确率大约99%
#   训练时，经常出现loss突然大幅增加的情况，是否fit内部采取了kick off优化？
# 重新开始训练，一次106 epoch，大约在40epoch模型就达到最优，准确率不到97%
#   大约在epoch 30附近有几次loss突然大幅增加，其它阶段大致平稳。
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=100, n_epoch=106, snapshot_step=False, callbacks=monitorCallback)

min(monitorCallback.iter)
max(monitorCallback.iter)
len(monitorCallback.iter)
monitorCallback.train_losses
monitorCallback.train_acc
monitorCallback.valid_losses
monitorCallback.valid_acc

# 将训练过程的 loss 和 acc 画出曲线
# 但是 tflearn 没有给出每一步的 验证集的loss和acc，没法画曲线
df_progress = pd.DataFrame(monitorCallback.iter, columns=['iter'])
df_progress['train_loss'] = monitorCallback.train_losses
df_progress['train_acc'] = monitorCallback.train_acc

#df_progress_valid = pd.DataFrame(monitorCallback.iter, columns=['iter'])
df_progress_valid = pd.DataFrame(monitorCallback.valid_losses, columns=['valid_losses'])
#df_progress_valid['valid_loss'] = monitorCallback.valid_losses
df_progress_valid['valid_acc'] = monitorCallback.valid_acc

df_progress.plot(x='iter', y='train_loss')
df_progress.plot(x='iter', y='train_acc')

df_progress_valid.plot(y='valid_losses')
df_progress_valid.plot(y='valid_acc')

# ## Testing
# After you're satisified with the training output and accuracy, you can then run the network on the **test data set** to measure it's performance! Remember, only do this after you've done the training and are satisfied with the results.
# 
# A good result will be **higher than 95% accuracy**. Some simple models have been known to get up to 99.7% accuracy!

# [ ]:

# Compare the labels that our model predicts with the actual labels

# Find the indices of the most confident prediction for each item. That tells us the predicted digit for that sample.
predictions = np.array(model.predict(testX)).argmax(axis=1)

# Calculate the accuracy, which is the percentage of times the predicated labels matched the actual labels
actual = testY.argmax(axis=1)
test_accuracy = np.mean(predictions == actual, axis=0)

# Print out the result
print("Test accuracy: ", test_accuracy)

#a=[1,2]
#b=[3,4]
#sum(np.array(a)*np.array(b))


