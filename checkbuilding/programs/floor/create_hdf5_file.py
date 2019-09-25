from keras import layers, models

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation="relu"))
model.add(layers.Dense(3,activation="sigmoid")) #分類先の種類分設定

# #モデル構成の確認
# model.summary()

from keras import optimizers

model.compile(loss="binary_crossentropy",
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=["acc"])

from keras.utils import np_utils
from functools import partial
import numpy as np

categories = ["2", "9", "10"]

nb_classes = len(categories)
np.load = partial(np.load, allow_pickle=True)
X_train, X_test, y_train, y_test = np.load("../../data/floor/uni_data.npy")

#データの正規化
X_train = X_train.astype("float") / 255
X_test = X_test.astype("float") / 255

#kerasで扱えるようにcategoriesをベクトルに変換
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

model = model.fit(X_train,
                  y_train,
                  epochs=10,
                  batch_size=7,
                  validation_data=(X_test, y_test))

import matplotlib.pyplot as plt

acc = model.history['acc']
val_acc = model.history['val_acc']
loss = model.history['loss']
val_loss = model.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('精度を示すグラフ')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('損失値を示すグラフ')

json_string = model.model.to_json()
open('./uni_predict.json', 'w').write(json_string)

#重みの保存

hdf5_file = "../../data/floor/uni_predict.hdf5"
model.model.save_weights(hdf5_file)
