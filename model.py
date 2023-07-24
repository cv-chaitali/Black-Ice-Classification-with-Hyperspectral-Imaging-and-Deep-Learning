import numpy as np
from keras.models import Model
from keras.layers import Input, Conv3D, Reshape, Conv2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

def build_model(input_shape, output_units):
    # Input layer
    input_layer = Input(shape=input_shape)

    # Convolutional layers
    conv_layer1 = Conv3D(filters=8, kernel_size=(2, 2, 3), activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(conv_layer1)
   

    print(conv_layer2.shape)
    conv3d_shape = conv_layer2.shape
    conv_layer3 = Reshape((conv3d_shape[1], conv3d_shape[2], conv3d_shape[3] * conv3d_shape[4]))(conv_layer2)
    conv_layer4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv_layer3)

    # Flattening layer
    flatten_layer = Flatten()(conv_layer4)

    # Fully connected layers
    dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)

    # Output layer
    output_layer = Dense(units=output_units, activation='softmax')(dense_layer2)

    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    return model

from data_proc import *
from data_vis import *

input_shape = (25, 25, 10, 1)
op_units = 10  

model = build_model(input_shape,op_units)

adam = Adam(learning_rate=0.001, decay=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

checkpoint = ModelCheckpoint(
    filepath=r'C:\Users10meters\BI.h5',
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

callbacks_list = [checkpoint]

#Xtrain and ytrain as training data and labels
history = model.fit(x=Xtrain, y=ytrain, batch_size=256, epochs=100, callbacks=callbacks_list)
