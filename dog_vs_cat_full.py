from keras.layers import Dropout,Dense,Flatten, Conv2D, Input, MaxPool2D, MaxPooling2D
from keras.models import Sequential, Model
import numpy as np
from keras.preprocessing.image import load_img, ImageDataGenerator, img_to_array
from keras.optimizers import RMSprop
from keras.losses import binary_crossentropy


train_dir = r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\data\train'
validate_dir = r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\data\validation'
file_output_pre_train = r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\data\pre_train.npy'
file_output_pre_validation = r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\data\pre_validation.npy'
file_save_weight  = r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\data\weight.h5'
new_weight  = r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\data\final_weight.h5'


number_train =2000
number_validate =800
width_size =150
height_size =150
batch_size = 16
epochs = 50



gen_train = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

data_gen = gen_train.flow_from_directory(
    directory=train_dir,
    target_size=(width_size,height_size),
    class_mode='binary',
    batch_size=batch_size,
)


gen_validation = ImageDataGenerator(rescale=1.0/255)

data_validation_gen = gen_validation.flow_from_directory(
    directory=validate_dir,
    target_size=(width_size,height_size),
    class_mode='binary',
    batch_size=batch_size,
)

# input_layer = Input(shape=(150,150,3))
model = Sequential()

model.add(Conv2D(
    filters=7,
    kernel_size=(3,3),
    strides=1,
    padding='same',
    input_shape=(150,150,3),
    activation='relu',

))
model.add(Conv2D(
    filters=7,
    kernel_size=(3, 3),
    strides=1,
    padding='same',
    input_shape=(150, 150, 3),
    activation='relu',
))
model.add(MaxPool2D(
    pool_size=(2,2),
    strides=2,
    padding='same'
))

model.add(Conv2D(
    filters=7,
    kernel_size=(3,3),
    strides=1,
    padding='same',
    input_shape=(150,150,3),
    activation='relu',

))
model.add(Conv2D(
    filters=7,
    kernel_size=(3, 3),
    strides=1,
    padding='same',
    input_shape=(150, 150, 3),
    activation='relu',
))

model.add(MaxPool2D(
    pool_size=(2,2),
    strides=2,
    padding='same'
))


model.add(Flatten())
model.add(Dense(
    units=256,
    activation='relu'
))
model.add(Dropout(0.5))
model.add(Dense(
    units=1,
    activation='sigmoid'
))


model.compile(
    optimizer=RMSprop(),
    loss= binary_crossentropy(),
    metrics=['accuracy'],
)

model.fit_generator(
    generator=data_gen,
    steps_per_epoch=number_train//batch_size,
    epochs = epochs,
    validation_data=data_validation_gen,
    validation_steps= number_validate//batch_size,
)
model.save_weights(filepath='')