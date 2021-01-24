from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

image_path = './images'
files = os.listdir(image_path)

dirs = []
for f in files:
    if os.path.isdir(os.path.join(image_path, f)):
        dirs.append(f)

print(dirs)

num_labels = len(dirs)

# 画像データのよみこみ
X_train, X_test, Y_train, Y_test = np.load(image_path + '/image_data.npy',
                                    allow_pickle=True)
# 整数から浮動小数に変換
X_train = X_train.astype('float') / 255.0
X_test = X_test.astype('float') / 255.0
# one hot vector に変換
Y_train = keras.utils.to_categorical(Y_train, num_labels)
Y_test = keras.utils.to_categorical(Y_test, num_labels)

# 学習モデルの作成
model = keras.models.Sequential()
model.add(layers.Conv2D(32, (3, 3), padding='same',
                        input_shape=X_train.shape[1:], activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))                        

model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_labels, activation='softmax'))

model.compile(loss='categorical_crossentropy',
            optimizer='RMSprop',
            metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=32, epochs=100)

model.save(image_path + '/cnn_data.h5')

# モデルの評価
scores = model.evaluate(X_test, Y_test, verbose=1)

print(scores)

