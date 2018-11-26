import numpy as np
import sys
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D, Dropout,BatchNormalization
from keras.utils import np_utils, plot_model
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard


if len(sys.argv)>1 :
	batch_size = int(sys.argv[1])
	num_epochs = int(sys.argv[2])
else:
	batch_size = 8
	num_epochs = 1
kernel_size = 3
kernel_size_1 = 1
pool_size =3
conv_depth_1 = 96
conv_depth_2 = 192
conv_depth_3 = 10
drop_prob_1 = 0.1
drop_prob_2 = 0.2
drop_prob_3 = 0.3
drop_prob_4 = 0.4
drop_prob_5 = 0.5

#tensorboard = TensorBoard(log_dir='./logs', write_graph=True)

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
drop_2 = Dropout(drop_prob_2)(pool_2)

conv_5 = Conv2D(conv_depth_2, kernel_size, kernel_size,  border_mode='same', activation='relu')(drop_2)
conv_5 = BatchNormalization(axis=1)(conv_5)

drop_3 = Dropout(drop_prob_3)(conv_5)

conv_6 = Conv2D(conv_depth_2, kernel_size_1, kernel_size_1,  border_mode='same', activation='relu')(drop_3)
conv_6 = BatchNormalization(axis=1)(conv_6)

drop_4 = Dropout(drop_prob_4)(conv_6)

conv_7 = Conv2D(conv_depth_3, kernel_size_1, kernel_size_1,  border_mode='same', activation='relu')(drop_4)
conv_7 = BatchNormalization(axis=1)(conv_7)

drop_5 = Dropout(drop_prob_5)(conv_7)


flat = Flatten()(drop_5)

out = Dense(num_classes,  activation='softmax')(flat)

model = Model(input=inp, output=out)
#plot_model(model, to_file='./graph_logs/model_structure.png', show_shapes=True)
print(model.summary())
input()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=num_epochs, verbose=1, validation_split=0.1
		  , callbacks=[EarlyStopping(monitor='val_loss', patience=1)])
#model.save ('/home/drew/PycharmProjects/cnn_cifar10/models/model.h5')
