import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input, Sequential

# Author: Haoming Zhang

def fcNN(datanum, channelnum=1):
  total_features = datanum * channelnum
  hidden_1 = min(3072, max(512, total_features // 2))
  hidden_2 = min(1536, max(256, total_features // 4))
  model = tf.keras.Sequential()
  model.add(Input(shape=(datanum, channelnum)))
  model.add(layers.Flatten())
  model.add(layers.Dense(hidden_1, activation=tf.nn.relu))
  model.add(layers.Dropout(0.3))


  model.add(layers.Dense(hidden_2))
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Dense(hidden_1))
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Dense(total_features))
  model.add(layers.Reshape((datanum, channelnum)))
  model.summary()
  return model


def RNN_lstm(datanum, channelnum=1):
  total_features = datanum * channelnum
  model = tf.keras.Sequential()
  model.add(Input(shape=(datanum,channelnum)))
  model.add(layers.LSTM(channelnum,return_sequences = True ))

  model.add(layers.Flatten())

  model.add(layers.Dense(total_features))
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Dense(total_features))
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Dense(total_features))
  model.add(layers.Reshape((datanum, channelnum)))
  model.summary()
  return model


def simple_CNN(datanum, channelnum=1):
  total_features = datanum * channelnum
  model = tf.keras.Sequential()

  model.add(layers.Conv1D(64, 3, strides=1, padding='same',input_shape=[ datanum, channelnum]))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv1D(64, 3, strides=1, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv1D(64, 3, strides=1, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  #num4
  model.add(layers.Conv1D(64, 3, strides=1, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Flatten())
  model.add(layers.Dense(total_features))
  model.add(layers.Reshape((datanum, channelnum)))

  model.build(input_shape=[ 1,datanum, channelnum] )
  model.summary()

  return model


# Resnet Basic Block module。
class Res_BasicBlock(layers.Layer):
  def __init__(self,kernelsize, stride=1):
    super(Res_BasicBlock, self).__init__()
    self.bblock = Sequential([layers.Conv1D(32,kernelsize,strides=stride,padding="same"),
                              layers.BatchNormalization(),
                              layers.ReLU(),
                              layers.Conv1D(16,kernelsize,strides=1,padding="same"),
                              layers.BatchNormalization(),
                              layers.ReLU(),
                              layers.Conv1D(32,kernelsize,strides=1,padding="same"),
                              layers.BatchNormalization(),
                              layers.ReLU()])
                              
    self.jump_layer = lambda x:x


  def call(self, inputs, training=None):

    #Through the convolutional layer
    out = self.bblock(inputs)

    #skip
    identity = self.jump_layer(inputs)

    output = layers.add([out, identity])  #layers下面有一个add，把这2个层添加进来相加。
    
    return output


class BasicBlockall(layers.Layer):
  def __init__(self, stride=1):
    super(BasicBlockall, self).__init__()

    self.bblock3 = Sequential([Res_BasicBlock(3),
                              Res_BasicBlock(3)
                              ])                      
    
    self.bblock5 = Sequential([Res_BasicBlock(5),
                              Res_BasicBlock(5)
                              ])                      

    self.bblock7 = Sequential([Res_BasicBlock(7),
                              Res_BasicBlock(7)
                              ])
                              
    self.downsample = lambda x:x


  def call(self, inputs, training=None):
 
    out3 = self.bblock3(inputs)
    out5 = self.bblock5(inputs)
    out7 = self.bblock7(inputs)

    out = tf.concat( values = [out3,out5,out7] , axis = -1)

    return out


def Complex_CNN(datanum, channelnum=1):
  total_features = datanum * channelnum
  model = Sequential()
  model.add(layers.Conv1D(32 ,5,strides=1,padding="same",input_shape=[ datanum, channelnum]))
  model.add(layers.BatchNormalization())
  model.add( layers.ReLU())

  model.add(BasicBlockall())

  model.add(layers.Conv1D(32 ,1,strides=1,padding="same"))
  model.add(layers.BatchNormalization())
  model.add( layers.ReLU())

  model.add(layers.Flatten())
  model.add(layers.Dense(total_features))
  model.add(layers.Reshape((datanum, channelnum)))

  model.build(input_shape=[ 1,datanum, channelnum] )
  model.summary()
  
  return model
