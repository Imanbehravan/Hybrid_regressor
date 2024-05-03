import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam


class deep_regressor():

    def __init__(self) -> None:
        self.model = Sequential()
        

    def fit(self, X_train, Y_train):
        input_layer = InputLayer(shape=(X_train.shape[1],))
        self.model.add(input_layer)
        self.model.add(Dense(5, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(2, activation='relu'))
        self.model.add(Dense(1))
        optimizer = Adam(learning_rate=0.001)
        self.model.compile(optimizer=optimizer, loss = 'mean_squared_error')
        self.model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

    def predict(self, X_test):
        pred = self.model.predict(np.array(X_test).reshape(1,-1))
        return pred

