from PIL import Image
import os, glob
import numpy as np

# NumPyの警告が出るのでそれを無視する
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# 画像が保存されているディレクトリのパス
image_path = './images'
files = os.listdir(image_path)

dirs = []
for f in files:
    if os.path.isdir(os.path.join(image_path, f)):
        dirs.append(f)

print(dirs)

image_size = 50
num_images = 200
num_testdata = 50

# 学習用画像とテスト画像を保存する配列を宣言
X_train = []
X_test = []
Y_train = []
Y_test = []

# ディレクトリでループして画像を読み込む
for index, label in enumerate(dirs):
    photos_dir = image_path + '/' + label
    # そのディレクトリ内のjpgファイルを全部読み込む
    photos = glob.glob(photos_dir + '/*.jpg')
    for i, p in enumerate(photos):
        if i >= num_images:
            break
        image = Image.open(p)
        image = image.convert('RGB')
        image = image.resize((image_size, image_size))
        data = np.asarray(image)

        if i < num_testdata:
            X_test.append(data)
            Y_test.append(index)
        else:
            for angle in range(-20, 21, 5):
                # 5度ずつ―20度から20度まで開店した画像を生成
                r_image = image.rotate(angle)
                data = np.asarray(r_image)
                X_train.append(data)
                Y_train.append(index)

                # 左右反転した画像を生成
                t_image = r_image.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(t_image)
                X_train.append(data)
                Y_train.append(index)

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

xy = (X_train, X_test, Y_train, Y_test)
print(len(X_train), len(X_test))
np.save(image_path + '/image_data.npy', xy)

