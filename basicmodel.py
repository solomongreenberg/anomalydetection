import keras as k
from keras.layers import Bidirectional, LSTM, Dense, Input
from keras.models import Model
from keras import optimizers
import tensorflow as tf
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

#from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from sklearn.metrics import mean_squared_log_error, mean_squared_error
import re
import datetime

from keras.models import *
from keras.layers import *
from keras.layers.core import Lambda
from keras.utils.vis_utils import plot_model
#import hdf5 as h5

riser_supply_file = 'data/01riser_supply_interp.pkl'
riser_return_file = 'data/01riser_return_interp.pkl'

with open(riser_supply_file, 'rb') as f:
    riser_supply = pkl.load(f)
with open(riser_return_file, 'rb') as f:
    riser_return = pkl.load(f)


lag_min = 0
lag_max = len(riser_supply[:,1])//2


#pts = []
#for lag in tqdm(range(lag_min, lag_max)):
#    n = len(riser_supply[:,1])
#    c1 = riser_supply[lag:,1]
#    c2 = riser_supply[:n-lag,1]
#    corr = np.corrcoef(c1, c2, ddof=0)[0, 1]
#    pts.append([lag, corr])
#
#pts = np.asarray(pts)
#pts[:,1] = pts[:,1] / np.max(pts[:,1])
#plt.plot(pts[:,0]/(24*60), pts[:,1])
#plt.show()

riser_supply[:,1] -= np.min(riser_supply[:,1])
#riser_supply[:,1] -= np.mean(riser_supply[:,1])
riser_supply[:,1] /= np.max(np.abs(riser_supply[:,1]))/2
riser_supply[:,1] -= 1

seq_length = 90 # 1.5 hour

x_train = []
y_train = []
invec_dim = 3
for i in range(0, len(riser_supply)-(seq_length+1)):
    x_train.append([
            riser_supply[i:i+seq_length,1], 
            np.sin(2*np.pi*riser_supply[i:i+seq_length,0]/(24*60*60)),
            np.cos(2*np.pi*riser_supply[i:i+seq_length,0]/(24*60*60))
            ])
    y_train.append(riser_supply[i+seq_length+1,1])

x_train = np.asarray(x_train).reshape((len(x_train), seq_length, invec_dim))
y_train = np.asarray(y_train)

test_prop = 0.05
test_number = int(round(x_train.shape[0]*test_prop))

x_test = x_train[-test_number:,:,:]
y_test = y_train[-test_number:]
x_train = x_train[:-test_number,:,:]
y_train = y_train[:-test_number]

x_train_quadrature = x_train[:,:,1:3]
x_train_temperature = x_train[:,:,0].reshape(x_train.shape[0], x_train.shape[1], 1)

input_temp = Input(shape=(x_train_temperature.shape[1], x_train_temperature.shape[2]))
input_time = Input(shape=(x_train_quadrature.shape[1], x_train_quadrature.shape[2]))

time_proc1 = Dense(48, activation='relu')(input_time)
time_proc2 = Dense(4, activation='relu')(time_proc1)

merged = concatenate([input_temp, time_proc2], axis=2)

lstm = Bidirectional(LSTM(64, activation='relu', return_sequences=True, dropout=0.3))(merged, training=True)
lstm = Bidirectional(LSTM(12, activation='relu', return_sequences=False, dropout=0.3))(lstm, training=True)
dense = Dense(32,activation='relu')(lstm)
out10 = Dense(1,activation='tanh')(dense)
out50 = Dense(1,activation='tanh')(dense)
out90 = Dense(1,activation='tanh')(dense)
model = Model([input_temp, input_time], [out10, out50, out90])

#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#import sys;sys.exit(0)

def q_loss(q, y, f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

losses = [
    lambda y,f: q_loss(0.1, y, f),
    lambda y,f: q_loss(0.5,y,f),
    lambda y,f:q_loss(0.9,y,f)
]

import os

if not os.path.exists('model.hdf5'):

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=losses, optimizer=adam, loss_weights=[0.3,0.3,0.3])
    
    
    history = model.fit([x_train_temperature, x_train_quadrature], [y_train,y_train,y_train], epochs=10, batch_size=seq_length*8, verbose=1, shuffle=True)# validation_data=(x_test, [y_test, y_test, y_test]))
    
    model.save_weights('model.hdf5')
    
print("-------EVAL-------")
    
from keras.models import load_model
model.load_weights('model.hdf5')

#pred_10, pred_50, pred_90 = [], [], []
#NN = K.function([model.layers[0].input, K.learning_phase()], 
#                [model.layers[-3].output,model.layers[-2].output,model.layers[-1].output])
#
#for i in tqdm.tqdm(range(0,100)):
#    predd = NN([x_test, 0.5])
#    pred_10.append(predd[0])
#    pred_50.append(predd[1])
#    pred_90.append(predd[2])
#
#pred_10 = np.asarray(pred_10)[:,:,0] 
#pred_50 = np.asarray(pred_50)[:,:,0]
#pred_90 = np.asarray(pred_90)[:,:,0]
#pred_90_m = np.exp(np.quantile(pred_90,0.9,axis=0) + init[5000:])
#pred_50_m = np.exp(pred_50.mean(axis=0) + init[5000:])
#pred_10_m = np.exp(np.quantile(pred_10,0.1,axis=0) + init[5000:])
#print(mean_squared_log_error(np.exp(y_test + init[5000:]), pred_50_m))
#

out = model.predict(x_test, verbose=1)
