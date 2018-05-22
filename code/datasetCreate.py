# 画像をnpz形式に変換する
# Create NPZ file
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
        img = img.resize((photo_size, photo_size))
        x.append(image_to_data(img))
        y.append(label)
        if not rotate: continue


def image_to_data(img): # 画像データを正規化
  data = np.asarray(img)
  data = data / 256
  data = data.reshape(photo_size, photo_size, 1) # RGBなら3を指定する
  return data

def make_dataset(max_photo, outfile, rotate):
    global x
    global y
    x = []
    y = []

    # folder_path = "./att_faces/"
    folders = glob.glob("./att_faces/*")
    for folder in folders:
        glob_images(folder,int(os.path.basename(folder)[1:]) - 1,max_photo,rotate)
    x = np.array(x, dtype=np.float32)
    np.savez(outfile, x=x, y=y)
    print("saved:" + outfile)

print("Create Start!!")
make_dataset(350, "photo.npz", rotate=True)
print("Success crate LearningData!")
make_dataset(50, "photo-test.npz", rotate=False)
print("Success create TestData!")

