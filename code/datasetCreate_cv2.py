# 画像をnpz形式に変換する
# Create NPZ file
import numpy as np
from PIL import Image
import os, glob, random, re, cv2

# 変数設定
photo_size = 92 # 画像サイズ
used_file = {}

def glob_images(path, label, max_photo, rotate):
    files = glob.glob(path + "/*.pgm")
    # 取り出した画像の順番をランダムに変更する
    random.shuffle(files)
    i = 0
    for f in files:
        if i >= max_photo:break
        if f in used_file:continue
        used_file[f] = True
        i += 1
        img = cv2.imread(f, 1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        x.append(image_to_data(img))
        y.append(label)
        
        if not rotate: continue
        # 以下テストデータ水増し用画像編集部分
        average(img, label)
        contrast(img, label)


# 平滑化
def average(img, label):
    for average in range(2, 6, 1):
        average_square = (average, average)
        blur_img = cv2.blur(img, average_square)
        x.append(image_to_data(img))
        y.append(label)
        average_square = (1, average)
        blur_img = cv2.blur(img, average_square)
        x.append(image_to_data(img))
        y.append(label)
        average_square = (average, 1)
        blur_img = cv2.blur(img, average_square)
        x.append(image_to_data(img))
        y.append(label)    # ルックアップテーブルの生成

# コントラスト調整
def contrast(img, label):
    # 以下データの水増し用画像変換
    min_table = 50
    max_table = 205
    diff_table = max_table - min_table

    LUT_HC = np.arange(256, dtype = 'uint8' )
    LUT_LC = np.arange(256, dtype = 'uint8' )

    # ハイコントラストLUT作成
    for i in range(0, min_table):
        LUT_HC[i] = 0
    for i in range(min_table, max_table):
        LUT_HC[i] = 255 * (i - min_table) / diff_table
    for i in range(max_table, 255):
        LUT_HC[i] = 255

    # ローコントラストLUT作成
    for i in range(256):
        LUT_LC[i] = min_table + i * (diff_table) / 255
    
    high_cont_img = cv2.LUT(img, LUT_HC)
    x.append(image_to_data(high_cont_img))
    y.append(label)
    low_cont_img = cv2.LUT(img, LUT_LC)
    x.append(image_to_data(low_cont_img))
    y.append(label)

    # 平滑化
    average(high_cont_img, label)
    average(low_cont_img, label)

def image_to_data(img): # 画像データを正規化
  data = np.asarray(img)
  data = img / 256
  data = data.reshape(92, 112, 1) # RGBなら3を指定する
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
make_dataset(8, "photo.npz", rotate=True)
print("Success crate LearningData!")
make_dataset(2, "photo-test.npz", rotate=False)
print("Success create TestData!")

