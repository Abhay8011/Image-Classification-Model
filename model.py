
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization

from keras.layers.advanced_activations import LeakyReLU


def load_model():

    model = Sequential()
    model.add(Conv2D(16, (3,3), input_shape=(128,128,3), padding='same'))
    model.add(LeakyReLU(0.5))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32, (3,3), padding='same'))
    model.add(LeakyReLU(0.5))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size= (2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(0.5))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(0.5))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256,activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(6))
    model.add(Activation("softmax"))

    return model

