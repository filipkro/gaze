from coordconv.coord import CoordinateChannel2D
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import activations
from tensorflow import keras

ip = Input(shape=(32, 32, 1))
#x = Flatten()(x)
#x = Softmax(axis=-1)(x)


x = CoordinateChannel2D()(ip)
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = BatchNormalization()(x)
x = CoordinateChannel2D()(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = BatchNormalization()(x)
x = Dense(512)(x)
x = Dense(512)(x)
x = Dropout(0.25)(x)
x = Dense(6)(x)
x = Activation(activations.linear)(x)

model = Model(ip, x)
model.summary()
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)

optimizer = Adam(lr=1e-2)
model.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy'])


"""
#create model
model = Sequential()
#add model layers
model.add()
model.add(Conv2D(64, kernel_size=3, activation=’relu’, input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation=’relu’))
model.add(Flatten())
model.add(Dense(10, activation=’softmax’))

model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
model.add(MaxPooling2D())
"""
