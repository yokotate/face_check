# 画像をnpz形式に変換する
import numpy as np
from PIL import Image
import os, glob, random, re

# 変数設定
photo_size = 92 # 画像サイズ
x = []
y = []
used_file = []

def glob_images(path, label, max_photo, rotate):
    files = glob.glob(path + "/*.pgm")
    # 取り出した画像の順番をランダムに変更する
    random.shuffle(files)
    i = 0
    for f in files:
        if i >= max_photo:break
        if f in used_file:continue
        i += 1
        img = Image.open(f)
        img = img.convert("L") # データをモノクロとして扱う。RGBも指定できる
        img = img.resize(photo_size, photo_size)
        x.append(image_to_data(img))
        y.append(label)
        # 回転させた画像データを使用することでテストデータの水増し
        for angle in range(-20 , 21 , 5):
            img_angle = img.rotate(angle)
            x.append(image_to_data(img_angle))
            y.append(label)


def image_to_data(img): # 画像データを正規化
  data = np.asarray(img)
  data = data / 256
  data = data.reshape(photo_size, photo_size, 2) # 白黒なので2を指定。RGBなら3
  return data

def make_dataset(max_photo, outfile, rotate):
    global x
    global y

    # folder_path = "./att_faces/"
    folders = glob.glob("./att_faces/*")
    for folder in folders:
        glob_images(folder,os.path.basename(folder),max_photo,rotate)
    x = np.array(x, dtype=np.float32)
    np.savez(outfile, x=x, y=y)
    print("saved:" + outfile)

make_dataset(350, "photo.npz", rotate=True)
make_dataset(50, "photo-test.npz", rotate=False)
