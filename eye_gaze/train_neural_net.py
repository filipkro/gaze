from coordconv.coord import CoordinateChannel2D

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import activations
from tensorflow import keras
import tfrecord_utils as utils
import tensorflow as tf
import numpy as np

def train(x_train,y_train):
    batch = 10
    ip = Input(shape=(32, 32, 1))


    x = CoordinateChannel2D()(ip)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = BatchNormalization()(x)
    x = CoordinateChannel2D(use_radius=True)(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(6, activation='linear')(x)
    #x = Activation(activations.linear)(x)

    model = Model(ip, x)
    model.summary()
    #keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)

    optimizer = Adam(lr=1e-2)
    model.compile(optimizer, loss='mse', metrics=['accuracy'])

    initial_learning_rate = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "Data/Models/CorCNN.h5", save_best_only=True
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )

    history = model.fit(x_train,y_train,epochs=15, validation_split=0.3)

    model.save('Data/Models/CorCNN.h5')



if __name__ == "__main__":
    
    npzfile = np.load("Data/generated_training.npz")
    sorted(npzfile.files)
    ['x', 'y']
    x = npzfile['x']
    y = npzfile['y']
    train(x,y)
