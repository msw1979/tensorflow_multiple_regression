# Author Dr. M. Alwarawrah
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import math, os, time, scipy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import tensorflow as tf
from IPython.display import Markdown, display
from tensorflow.keras.layers import Flatten
from sklearn.model_selection import train_test_split

print('Tensorflow version', tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

# start recording time
t_initial = time.time()

#compute accuracy and percent error 
def accu(y_pred,y_test):
    accuracy = (1- np.mean(abs(y_pred-y_test)/y_test))*100
    PE = np.mean(abs(y_pred-y_test)/y_test)*100
    return accuracy, PE

# Plot the loss
def plot_loss(train_loss, val_loss, name):
    plt.clf()
    plt.plot(train_loss, label='Training', color='k')
    plt.plot(val_loss, label='Validation', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_vs_epoch_%s.png'%(name))

#plot prediction
def plot_predicition(x_test, y_test, y_pred, feaature_name, target_name, name):
    for i in range(0,len(feaature_name)):
        plt.clf()
        plt.scatter(x_test[:,i], y_test, label='Data', color='red')
        plt.scatter(x_test[:,i], y_pred, color='k', label='Predictions')
        plt.xlabel('%s'%feaature_name[i])
        plt.ylabel('%s'%target_name)
        plt.legend()
        plt.savefig('prediction_%s_%s.png'%(feaature_name[i], name))

#Define a Dense layer class
class Dense_layer(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Dense_layer, self).__init__()
        self.units = units
    #initialize w and b
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),initializer="random_normal",trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

#Define Convolutional Neural Network Model
class multiple_reg(tf.keras.Model):
    def __init__(self):
        super(multiple_reg, self).__init__()
        #flatten the tensor shape
        self.normalize = tf.keras.layers.Normalization()
        #dense layer 1 and 2
        self.Dense1 = Dense_layer(10)
        self.Dense2 = Dense_layer(10)
        self.Dense3 = Dense_layer(1)

    def call(self, x):
        x = self.normalize(x)
        x = self.Dense1(x)
        x = tf.nn.relu(x)
        x = self.Dense2(x)
        x = tf.nn.relu(x)
        x = self.Dense3(x)
        return x

#train model
#return train/test loss and accuracy
def train_model(model, train_ds, x_test, y_test, criterion, optimizer, epochs, x_names, y_name, output_file):
    #define list
    train_loss=[]
    train_acc = []
    train_PE = []
    test_loss = []
    test_acc = []
    test_PE = []

    for epoch in range(epochs):
        #training part
        #define list for loss and accuracy batches
        batch_losses_train = []
        batch_acc_train = []
        batch_PE_train = []
        for x_train_batch, y_train_batch in train_ds:
            with tf.GradientTape() as tape:
                #prediction
                z = model(x_train_batch)
                #compute loss
                loss = criterion(y_train_batch, z)
                #append loss
                batch_losses_train.append(loss.numpy())
            #compute gradient    
            grads = tape.gradient(loss, model.trainable_variables)
            #optimize the variables
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            #calculate accuracy & PE
            accuracy, PE = accu(z.numpy(), y_train_batch.numpy())
            #append the accuracy and PE
            batch_acc_train.append(accuracy)
            batch_PE_train.append(PE)
        #  append loss and accuracy (need to average)
        train_loss.append(tf.reduce_sum(batch_losses_train))
        train_acc.append(tf.reduce_mean(batch_acc_train))
        train_PE.append(tf.reduce_mean(batch_PE_train))

        #print train results on screen and file
        print("Class Model, Training, Epoch: {}, Loss: {:.2f}, Accuracy: {:.2f}, PE: {:.2f}".format(epoch, tf.reduce_mean(batch_losses_train), tf.reduce_mean(batch_acc_train), tf.reduce_mean(batch_PE_train)) , file=output_file) 
        print("Class Model, Training, Epoch: {}, Loss: {:.2f}, Accuracy: {:.2f}, PE: {:.2f}".format(epoch, tf.reduce_mean(batch_losses_train), tf.reduce_mean(batch_acc_train), tf.reduce_mean(batch_PE_train))) 

        #validation
        N_test = len(x_test)
        #get prediction
        z = model( x_test )
        # calculate loss
        loss = criterion( y_test, z).numpy()
        #append loss
        test_loss.append(loss)
        # calculate accuracy and PE
        accuracy, PE = accu(z, y_test)
        #append accuracy
        test_acc.append(accuracy)
        test_PE.append(PE)

        #print train results on screen and file
        print("Class Model, Validation, Epoch: {}, Loss: {:.2f}, Accuracy: {:.2f}, PE: {:.2f}".format(epoch, loss, accuracy, PE) , file=output_file) 
        print("Class Model, Validation, Epoch: {}, Loss: {:.2f}, Accuracy: {:.2f}, PE: {:.2f}".format(epoch, loss, accuracy, PE)) 

    #plot predition
    y_pred_test = model( x_test )
    plot_predicition(x_test, y_test, y_pred_test, x_names, y_name, 'class_model')

    #return train/test loss and  accuracy
    return train_loss, train_acc, test_loss, test_acc

#plot Loss and Accuracy vs epoch
def plot_loss_accuracy(train_loss, train_acc, val_loss, val_acc, name):
    plt.clf()
    fig,ax = plt.subplots()
    ax.plot(train_loss, color='k', label = 'Training Loss')
    ax.plot(val_loss, color='r', label = 'Validation Loss')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=16)
    ax2 = ax.twinx()
    ax2.plot(train_acc, color='b', label = 'Training Accuracy')
    ax2.plot(val_acc, color='g', label = 'Validation Accuracy')
    ax2.set_ylabel('Accuracy', fontsize=16)
    fig.legend(loc ="center")
    fig.tight_layout()
    plt.savefig('loss_accuracy_epoch_%s.png'%(name))

#open file
output_file = open('output.txt','w')

#execut eagerly
eagerly_decision = tf.executing_eagerly()
print('executing Eagerly: {}'.format(eagerly_decision), file=output_file)
print('executing Eagerly: {}'.format(eagerly_decision))


#define columns names list
col_names = ["Make","Model","Vehicle_Class","Engine_Size","Cylinders","Transmission","Fuel_Type","Fuel_Consumption_City","Fuel_Consumption_Hwy","Fuel_Consumption_Comb","Fuel_Consumption_Comb_mpg","CO2_Emissions"]

#Read dataframe and skip first raw that contain header
df = pd.read_csv('CO2 Emissions_Canada.csv',names=col_names, header = None, skiprows = 1)

#define features and target dataframe
features = np.asanyarray(df[["Engine_Size","Cylinders","Fuel_Consumption_Comb"]])
target = np.asanyarray(df[["CO2_Emissions"]])

#split features and target to train and test data sets
x_train, x_test, y_train, y_test = train_test_split( features, target, test_size=0.2, random_state=4)

#define a names list for features and 
x_names = df[["Engine_Size","Cylinders","Fuel_Consumption_Comb"]].columns.to_list()
y_name = df[["CO2_Emissions"]].columns.to_list()

# define number of epochs, learning rate and batch size
epochs = 20
learning_rate = 0.01
batch_size = 50

#define criterion to calculate loss
criterion = tf.keras.losses.MeanAbsoluteError()

#1) using Sequential model
#------SEQ Model-----#
model_seq = tf.keras.Sequential([tf.keras.layers.Normalization(axis=-1),
                                 tf.keras.layers.Dense(10, activation='relu'),
                                 tf.keras.layers.Dense(10, activation='relu'),
                                 tf.keras.layers.Dense(1)])

#define optimizer
optimizer_seq = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#compile model
model_seq.compile(optimizer=optimizer_seq,loss=criterion, metrics=['accuracy'])

#training results
results = model_seq.fit(x_train,y_train, batch_size = batch_size, epochs=epochs, validation_data=(x_test, y_test))
train_loss = results.history['loss']
val_loss = results.history['val_loss']

#predicition
y_pred_test = model_seq.predict(x_test)
y_pred_train = model_seq.predict(x_train)

#print accuracy
print('SEQ Model, Training, Loss: %5.3f'%(train_loss[-1]) ,'Accuracy: %5.2f'%accu(y_pred_train,y_train)[0],'%', 'Percent Error: %5.2f'%accu(y_pred_train,y_train)[1], '%')
print('SEQ Model, Training, Loss: %5.3f'%(train_loss[-1]) ,'Accuracy: %5.2f'%accu(y_pred_train,y_train)[0],'%', 'Percent Error: %5.2f'%accu(y_pred_train,y_train)[1], '%', file=output_file)
print('SEQ Model, Validation, Loss: %5.3f'%(val_loss[-1]) ,'Accuracy: %5.2f'%accu(y_pred_test,y_test)[0],'%', 'Percent Error: %5.2f'%accu(y_pred_test,y_test)[1], '%')
print('SEQ Model, Validation, Loss: %5.3f'%(val_loss[-1]) ,'Accuracy: %5.2f'%accu(y_pred_test,y_test)[0],'%', 'Percent Error: %5.2f'%accu(y_pred_test,y_test)[1], '%', file=output_file)

#plot loss for training and validation
plot_loss(train_loss, val_loss, 'seq_model')

#plot predition
plot_predicition(x_test, y_test, y_pred_test, x_names, y_name, 'seq_model')

#------SEQ Model END-----#

# 2) using Customed class model
#-------Class model--------------#
#train model and get train and validation loss and accuracy
model_class =  multiple_reg()

#slice training data to batches. You can do the samething to test data
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

#define optimizer
optimizer_class = tf.keras.optimizers.Adam(learning_rate=learning_rate)

#train the model using the function train_model
train_loss, train_acc, val_loss, val_acc = train_model(model_class, train_ds, x_test, y_test, criterion, optimizer_class, epochs,x_names,y_name, output_file)

#plot loss for training and validation
plot_loss_accuracy(train_loss, train_acc, val_loss, val_acc, 'model_class')
#------Class Model END-----#

output_file.close()

#End recording time
t_final = time.time()

t_elapsed = t_final - t_initial
hour = int(t_elapsed/(60.0*60.0))
minute = int(t_elapsed%(60.0*60.0)/(60.0))
second = t_elapsed%(60.0)
print("%d h: %d min: %f s"%(hour,minute,second))