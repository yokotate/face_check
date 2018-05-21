import numpy as np
import glob
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from tensorflow.python.keras.callbacks import TensorBoard


data = np.load("./photo.npz")
x_train = data["x"]
y_train = data["y"]
data = np.load("./photo-test.npz")
x_test = data["x"]
y_test = data["y"]
classes = len(glob.glob("./att_faces/*"))

x_train = x_train / 255.
x_test = x_test / 255.

y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

model = Sequential()


model.add(
    Conv2D(
        filters=92,
        input_shape=(92,92,1),
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'
    )
)
model.add(
    Conv2D(
        filters=92,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'
    )
)

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))
model.add(
    Conv2D(
        filters=184,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'
    )
)
model.add(
    Conv2D(
        filters=184,
        kernel_size=(3,3),
        strides=(1,1),
        padding='same',
        activation='relu'
    )
)
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.output_shape

model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=classes, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
tsb=TensorBoard(log_dir='./logs')
history_model = model.fit(
    x_train,
    y_train,
    batch_size=92,
    epochs=20,
    validation_split=0.2,
    callbacks=[tsb]
)
