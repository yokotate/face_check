import numpy as np
import glob
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from keras.callbacks import TensorBoard

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
        input_shape=(112,92,1),
        kernel_size=(3, 3),
        strides=(2,2),
        padding='same',
        activation='relu'
    )
)
model.add(
    Conv2D(
        filters=92,
        kernel_size=(3, 3),
        strides=(2,2),
        padding='same',
        activation='relu'
    )
)

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(
    Conv2D(
        filters=184,
        kernel_size=(3, 3),
        strides=(2,2),
        padding='same',
        activation='relu'
    )
)
model.add(
    Conv2D(
        filters=184,
        kernel_size=(3, 3),
        strides=(2,2),
        padding='same',
        activation='relu'
    )
)

model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
model.add(Dropout(0.25))

model.add(Flatten())
model.output_shape

model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.25))
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
    batch_size=100,
    epochs=50,
    validation_split=0.25,
    callbacks=[tsb]
)
model.save_weights("nn.hdf5")
score = model.evaluate(x_test, y_test)
print('loss = ', score[0])
print('accuracy = ', score[1])

