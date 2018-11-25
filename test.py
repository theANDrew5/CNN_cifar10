import numpy as np
import  sys

from keras.datasets import cifar10
from keras.models import Model, load_model
from keras.utils import np_utils, plot_model



(X_train, y_train), (X_test, y_test) = cifar10.load_data()
num_train, depth, height, width = X_train.shape
num_test = X_test.shape[0]
num_classes = np.unique(y_test).shape[0]

X_test = X_test.astype('float32')

X_test /= np.max(X_train)

Y_test = np_utils.to_categorical(y_test, num_classes)


model = load_model(str(sys.argv[1]))
plot_model(model, to_file='./graph_logs/model_structure.png', show_shapes=True)
print(model.summary())
input()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = model.evaluate(X_test, Y_test, verbose=1)
print(score)
