
# coding: utf-8

# # 你的第一个神经网络
# 
# 在此项目中，你将构建你的第一个神经网络，并用该网络预测每日自行车租客人数。我们提供了一些代码，但是需要你来实现神经网络（大部分内容）。提交此项目后，欢迎进一步探索该数据和模型。



#get_ipython().magic('matplotlib inline')
#get_ipython().magic("config InlineBackend.figure_format = 'retina'")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## 加载和准备数据
# 
# 构建神经网络的关键一步是正确地准备数据。不同尺度级别的变量使网络难以高效地掌握正确的权重。我们在下方已经提供了加载和准备数据的代码。你很快将进一步学习这些代码！



data_path = 'Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)



rides.head()


# ## 数据简介
# 
# 此数据集包含的是从 2011 年 1 月 1 日到 2012 年 12 月 31 日期间每天每小时的骑车人数。骑车用户分成临时用户和注册用户，cnt 列是骑车用户数汇总列。你可以在上方看到前几行数据。
# 
# 下图展示的是数据集中前 10 天左右的骑车人数（某些天不一定是 24 个条目，所以不是精确的 10 天）。你可以在这里看到每小时租金。这些数据很复杂！周末的骑行人数少些，工作日上下班期间是骑行高峰期。我们还可以从上方的数据中看到温度、湿度和风速信息，所有这些信息都会影响骑行人数。你需要用你的模型展示所有这些数据。



rides[:24*10].plot(x='dteday', y='cnt', figsize=(10,4))

# 查看每天的骑行数据，对比2011年和2012年
day_rides = pd.read_csv('Bike-Sharing-Dataset/day.csv')
day_rides = day_rides.set_index(['dteday'])

day_rides.loc['2011-10-01':'2011-12-31'].plot(y='cnt', figsize=(10,4))
day_rides.loc['2012-10-01':'2012-12-31'].plot(y='cnt', figsize=(10,4))

day_rides.loc['2011-10-28':'2011-11-03']
day_rides.loc['2012-10-28':'2012-11-03']

# ### 虚拟变量（哑变量）
# 
# 下面是一些分类变量，例如季节、天气、月份。要在我们的模型中包含这些数据，我们需要创建二进制虚拟变量。用 Pandas 库中的 `get_dummies()` 就可以轻松实现。

# [5]:

dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
data.head()




quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std


# ### 调整目标变量
# 
# 为了更轻松地训练网络，我们将对每个连续变量标准化，即转换和调整变量，使它们的均值为 0，标准差为 1。
# 
# 我们会保存换算因子，以便当我们使用网络进行预测时可以还原数据。

# ### 将数据拆分为训练、测试和验证数据集
# 
# 我们将大约最后 21 天的数据保存为测试数据集，这些数据集会在训练完网络后使用。我们将使用该数据集进行预测，并与实际的骑行人数进行对比。



# Save data for approximately the last 21 days 
test_data = data[-21*24:]

# Now remove the test data from the data set 
data = data[:-21*24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]


# 我们将数据拆分为两个数据集，一个用作训练，一个在网络训练完后用来验证网络。因为数据是有时间序列特性的，所以我们用历史数据进行训练，然后尝试预测未来数据（验证数据集）。

# [8]:

# Hold out the last 60 days or so of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]


# ## 开始构建网络
# 
# 下面你将构建自己的网络。我们已经构建好结构和反向传递部分。你将实现网络的前向传递部分。还需要设置超参数：学习速率、隐藏单元的数量，以及训练传递数量。
# 
# <img src="assets/neural_network.png" width=300px>
# 
# 该网络有两个层级，一个隐藏层和一个输出层。隐藏层级将使用 S 型函数作为激活函数。输出层只有一个节点，用于递归，节点的输出和节点的输入相同。即激活函数是 $f(x)=x$。这种函数获得输入信号，并生成输出信号，但是会考虑阈值，称为激活函数。我们完成网络的每个层级，并计算每个神经元的输出。一个层级的所有输出变成下一层级神经元的输入。这一流程叫做前向传播（forward propagation）。
# 
# 我们在神经网络中使用权重将信号从输入层传播到输出层。我们还使用权重将错误从输出层传播回网络，以便更新权重。这叫做反向传播（backpropagation）。
# 
# > **提示**：你需要为反向传播实现计算输出激活函数 ($f(x) = x$) 的导数。如果你不熟悉微积分，其实该函数就等同于等式 $y = x$。该等式的斜率是多少？也就是导数 $f(x)$。
# 
# 
# 你需要完成以下任务：
# 
# 1. 实现 S 型激活函数。将 `__init__` 中的 `self.activation_function`  设为你的 S 型函数。
# 2. 在 `train` 方法中实现前向传递。
# 3. 在 `train` 方法中实现反向传播算法，包括计算输出错误。
# 4. 在 `run` 方法中实现前向传递。
# 
#   

# [48]:

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
#        
#        def sigmoid(x):
#            return 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation here
#        self.activation_function = sigmoid
                    
    
    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        #print('features',features)
        #print('targets',targets)
#        nCount = 0
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
#            nCount += 1
#            if(nCount > 1):
#                break
            #print('#######################################')
            #### Implement the forward pass here ####
            ### Forward pass ###
            # TODO: Hidden layer - Replace these values with your calculations.
            #print('X.shape',X.shape)
            #print('X',X)
            #print('X[None,:].shape', X[None,:].shape)
            #print('X[None,:]', X[None,:])
            #print('y.shape',y.shape)
            #print('y',y)
            #print('weights_input_to_hidden.shape', self.weights_input_to_hidden.shape)
            #print('weights_hidden_to_output.shape', self.weights_hidden_to_output.shape)
            hidden_inputs = np.matmul(X[None,:], self.weights_input_to_hidden) # signals into hidden layer
            #print('hidden_inputs.shape', hidden_inputs.shape)
            hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
            #print('hidden_outputs.shape',hidden_outputs.shape)

            # TODO: Output layer - Replace these values with your calculations.
            final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
            final_outputs = final_inputs # signals from final output layer
            #print('final_inputs.shape', final_inputs.shape)
            #print('final_outputs.shape', final_outputs.shape)
            
            #### Implement the backward pass here ####
            ### Backward pass ###

            # TODO: Output error - Replace this value with your calculations.
            error = y - final_outputs # Output layer error is the difference between desired target and actual output.
            #print('error.shape', error.shape)
            #print('y',y)
            #print('final_outputs',final_outputs)
            #print('error',error)
            output_error_term = error * 1
            #print('output_error_term',output_error_term)
            
            # TODO: Calculate the hidden layer's contribution to the error
            hidden_error = np.matmul(output_error_term, self.weights_hidden_to_output.T)
#            hidden_error = output_error_term * self.weights_hidden_to_output.T
            #print('hidden_error.shape',hidden_error.shape)
#            print('hidden_error1',hidden_error1)
#            print('hidden_error',hidden_error)
            
            # TODO: Backpropagated error terms - Replace these values with your calculations.
            #output_error_term = None
            hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
            #print('hidden_error_term.shape', hidden_error_term.shape)

            # Weight step (input to hidden)
            tmp = np.matmul(X[:,None], hidden_error_term)
            #print('X[:,None]',X[:,None])
            #print('hidden_error_term',hidden_error_term)
            #print('tmp.shape',tmp.shape)
            #print('tmp',tmp)
            delta_weights_i_h += tmp
            #print('delta_weights_i_h.shape',delta_weights_i_h.shape)
            #print('-------------------')
            # Weight step (hidden to output)
            #print('hidden_outputs', hidden_outputs)
            #print('output_error_term', output_error_term)
            tmp = hidden_outputs.T * output_error_term
            #print('tmp.shape', tmp.shape)
            #print('tmp', tmp)
            delta_weights_h_o += tmp
            #print('delta_weights_h_o.shape', delta_weights_h_o.shape)

        # TODO: Update the weights - Replace these values with your calculations.
        #print('self.lr, n_records',self.lr, n_records)
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        #print('self.weights_hidden_to_output', self.weights_hidden_to_output)
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step
        #print('self.weights_input_to_hidden', self.weights_input_to_hidden)
        
    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
#        #print('features.shape', features.shape)
#        #print(features)
        hidden_inputs = np.matmul(features, self.weights_input_to_hidden) # signals into hidden layer
#        #print('hedden_inputs', hidden_inputs)
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
#        #print('hidden_outputs', hidden_outputs)
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
#        #print('final_inputs', final_inputs)
        final_outputs = final_inputs # signals from final output layer 
#        #print('final_outputs', final_outputs)
        
        return final_outputs


# [49]:

def MSE(y, Y):
    return np.mean((y-Y)**2)


# ## 单元测试
# 
# 运行这些单元测试，检查你的网络实现是否正确。这样可以帮助你确保网络已正确实现，然后再开始训练网络。这些测试必须成功才能通过此项目。

# [50]:

import unittest

inputs = np.array([[0.5, -0.2, 0.1]])
targets = np.array([[0.4]])
test_w_i_h = np.array([[0.1, -0.2],
                       [0.4, 0.5],
                       [-0.3, 0.2]])
test_w_h_o = np.array([[0.3],
                       [-0.1]])

class TestMethods(unittest.TestCase):
    
    ##########
    # Unit tests for data loading
    ##########
    
    def test_data_path(self):
        # Test that file path to dataset has been unaltered
        self.assertTrue(data_path.lower() == 'bike-sharing-dataset/hour.csv')
        
    def test_data_loaded(self):
        # Test that data frame loaded
        self.assertTrue(isinstance(rides, pd.DataFrame))
    
    ##########
    # Unit tests for network functionality
    ##########

    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        # Test that the activation function is a sigmoid
        self.assertTrue(np.all(network.activation_function(0.5) == 1/(1+np.exp(-0.5))))
        self.assertTrue(np.all(network.activation_function(0.1) == 1/(1+np.exp(-0.1))))

    def test_train(self):
        # Test that weights are updated correctly on training
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()
        
        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_hidden_to_output, 
                                    np.array([[ 0.37275328], 
                                              [-0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[ 0.10562014, -0.20185996], 
                                              [0.39775194, 0.50074398], 
                                              [-0.29887597, 0.19962801]])))

    def test_run(self):
        # Test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

#        #print('inputs.shape', inputs.shape)
        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))

suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
unittest.TextTestRunner().run(suite)

#import os
#os.system("pause")

# ## 训练网络
# 
# 现在你将设置网络的超参数。策略是设置的超参数使训练集上的错误很小但是数据不会过拟合。如果网络训练时间太长，或者有太多的隐藏节点，可能就会过于针对特定训练集，无法泛化到验证数据集。即当训练集的损失降低时，验证集的损失将开始增大。
# 
# 你还将采用随机梯度下降 (SGD) 方法训练网络。对于每次训练，都获取随机样本数据，而不是整个数据集。与普通梯度下降相比，训练次数要更多，但是每次时间更短。这样的话，网络训练效率更高。稍后你将详细了解 SGD。
# 
# 
# ### 选择迭代次数
# 
# 也就是训练网络时从训练数据中抽样的批次数量。迭代次数越多，模型就与数据越拟合。但是，如果迭代次数太多，模型就无法很好地泛化到其他数据，这叫做过拟合。你需要选择一个使训练损失很低并且验证损失保持中等水平的数字。当你开始过拟合时，你会发现训练损失继续下降，但是验证损失开始上升。
# 
# ### 选择学习速率
# 
# 速率可以调整权重更新幅度。如果速率太大，权重就会太大，导致网络无法与数据相拟合。建议从 0.1 开始。如果网络在与数据拟合时遇到问题，尝试降低学习速率。注意，学习速率越低，权重更新的步长就越小，神经网络收敛的时间就越长。
# 
# 
# ### 选择隐藏节点数量
# 
# 隐藏节点越多，模型的预测结果就越准确。尝试不同的隐藏节点的数量，看看对性能有何影响。你可以查看损失字典，寻找网络性能指标。如果隐藏单元的数量太少，那么模型就没有足够的空间进行学习，如果太多，则学习方向就有太多的选择。选择隐藏单元数量的技巧在于找到合适的平衡点。

# [ ]:

import sys

### Set the hyperparameters here ###
iterations = 10000
learning_rate = 0.15
hidden_nodes = 20
output_nodes = 1

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for ii in range(iterations):
    #print('===============================================')
    #print('===============================================')
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    X, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']
                             
    network.train(X, y)
    
    # Printing out the training progress
    train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations))                      + "% ... Training loss: " + str(train_loss)[:5]                      + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)


# [ ]:
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(losses['train'], label='Training loss')
ax.plot(losses['validation'], label='Validation loss')
ax.legend()
_ = plt.ylim()


# ## 检查预测结果
# 
# 使用测试数据看看网络对数据建模的效果如何。如果完全错了，请确保网络中的每步都正确实现。

# [ ]:

fig, ax = plt.subplots(figsize=(8,4))

mean, std = scaled_features['cnt']
predictions = network.run(test_features).T*std + mean
ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)


# ## 可选：思考下你的结果（我们不会评估这道题的答案）
# 
#  
# 请针对你的结果回答以下问题。模型对数据的预测效果如何？哪里出现问题了？为何出现问题呢？
# 
# > **注意**：你可以通过双击该单元编辑文本。如果想要预览文本，请按 Control + Enter
# 
# #### 请将你的答案填写在下方
# 
#
# 预测结果与实际数据大致吻合。比较大的差异出现在12月下旬，有可能跟圣诞节有关。圣诞节每年一次，但这里只有2年的数据，所以这种跟年度相关的特征难以被模型学会。虽然数据中有holiday字段，但对于圣诞节也只有25号当天其holiday=1，难以体现这一重大节日的影响。如果有多年的数据可能提升性能，或者改进holiday字段可能也有帮助。
