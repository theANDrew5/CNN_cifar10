import numpy as np
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D, Dropout,BatchNormalization
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

batch_size = 16
num_epochs = 10
kernel_size = 3
pool_size =2
conv_depth_1 = 16
conv_depth_2 = 32
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 512
hidden_size_1=256

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

num_train, depth, height, width = X_train.shape
num_test = X_test.shape[0]
num_classes = np.unique(y_train).shape[0]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= np.max(X_train)
X_test /= np.max(X_train)

Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

inp = Input(shape=(depth, height, width))
inp_norm = BatchNormalization(axis=1)(inp)

conv_1 = Conv2D(conv_depth_1, kernel_size, kernel_size,  border_mode='same', activation='relu')(inp_norm)
conv_1 = BatchNormalization(axis=1)(conv_1)
conv_2 = Conv2D(conv_depth_1, kernel_size, kernel_size,  border_mode='same', activation='relu')(conv_1)
conv_2 = BatchNormalization(axis=1)(conv_2)

pool_1 = MaxPool2D(pool_size=(pool_size,pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)

conv_3 = Conv2D(conv_depth_2, kernel_size, kernel_size,  border_mode='same', activation='relu')(drop_1)
conv_3 = BatchNormalization(axis=1)(conv_3)
conv_4 = Conv2D(conv_depth_2, kernel_size, kernel_size,  border_mode='same', activation='relu')(conv_3)
conv_4 = BatchNormalization(axis=1)(conv_4)

pool_2 = MaxPool2D(pool_size=(pool_size,pool_size))(conv_4)
drop_2 = Dropout(drop_prob_1)(pool_2)

flat = Flatten()(drop_2)

hidden_1 = Dense(hidden_size,  activation='relu')(flat)
hidden_1 = BatchNormalization(axis=1)(hidden_1)

drop_3 = Dropout(drop_prob_2)(hidden_1)

hidden_2 = Dense(hidden_size_1, activation='relu')(drop_3)
hidden_2 = BatchNormalization(axis=1)(hidden_2)

drop_4 = Dropout(drop_prob_2)(hidden_2)

out = Dense(num_classes,  activation='softmax')(drop_4)

model = Model(input=inp, output=out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=num_epochs, verbose=1, validation_split=0.1
		  , callbacks=[EarlyStopping(monitor='val_loss', patience=1)])
model.evaluate(X_test, Y_test, verbose=1)