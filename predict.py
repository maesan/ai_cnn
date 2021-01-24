from tensorflow import keras
import numpy as np
import os
from PIL import Image

# 画像が保存されているディレクトリのパス
image_path = './images'
files = os.listdir(image_path)

dirs = []
for f in files:
    if os.path.isdir(os.path.join(image_path, f)):
        dirs.append(f)

print(dirs)

num_labels = len(dirs)
image_size = 50

model = keras.models.load_model(image_path + '/cnn_data.h5')

while True:
    print('input image file: ', end='')
    filename = input()

    image = Image.open(filename).convert('RGB').resize((image_size, image_size))
    data = np.asarray(image)
    data = data.astype('float') / 255.0
    x = np.array([data,])

    result = model.predict(x)
    print(result)
    print(dirs[np.argmax(result)])

