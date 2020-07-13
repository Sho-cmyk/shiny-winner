from PIL import Image
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

classes = ["ウソップ", "サンジ", "ジンベエ", "ゾロ", "チョッパー",
           "ナミ", "ビビ", "フランキー", "ブルック", "ルフィ", "ロビン"]
num_classes = len(classes)
image_size = 50
num_testdata = 30

# 画像の読み込み
X_train = []
X_test = []
Y_train = []
Y_test = []

for index, classlabel in enumerate(classes):
    photos_dir = "./"+classlabel
    files = glob.glob(photos_dir+"/*.jpg")
    for i, file in enumerate(files):
        if i >= 200:
            break
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X_train.append(data)  # リストの最後尾に追加する
        Y_train.append(index)  # ラベルを追加する

datagen = ImageDataGenerator(
    samplewise_center=True, samplewise_std_normalization=True)


g = datagen.flow(X_train, Y_train, shuffle=False)
X_batch, y_batch = g.next()

X_batch *= 127.0/max(abs(X_batch.min()), X_batch.max())
X_batch += 127.0
X_batch = X_batch.astype('unit8')

# 配列にしてテストとトレーニングデータに分けて入れる
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(Y_train)
y_test = np.array(Y_test)

# X_train,X_test,y_train,y_test=train_test_split(X,Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./one-piece_aug.npy", xy)  # コードを保存
