import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout,Flatten
from keras.models import Model, Sequential
import numpy as np
from keras.applications import VGG16
from keras.activations import relu,sigmoid
from keras.optimizers import SGD
from keras.losses import binary_crossentropy

train_dir = r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\data\train'
validate_dir = r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\data\validation'
file_output_pre_train = r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\data\pre_train.npy'
file_output_pre_validate = r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\data\pre_validation.npy'
file_save_weight  = r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\learn_five9\data\weight.h5'

number_train =2000
number_validate =800
width_size =150
height_size =150
batch_size = 16
epochs = 100

def create_input():
    model = VGG16(include_top=False,weights='imagenet')
    # có thể thêm normalize, augmentation image
    gen_data  = ImageDataGenerator(
        rescale=1.0/255,
    )

    # train_generator là 1 generator: muốn xem từng thành phần thì ta dùng .next, mỗi thành phần là 1 tập batch bao gồm: X_batch, y_batch với X_batch có shape (batch_size, widht_size, height_size, number_color)
    # thành phần thứ 2 tương ứng là label của mỗi input, tùy vào parameter class_mode
    train_generator = gen_data.flow_from_directory(
        directory=train_dir,
        target_size= (width_size,height_size),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False # để lát cón lấy label
    )
    validate_generator = gen_data.flow_from_directory(
        directory=validate_dir,
        target_size=(width_size, height_size),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )

    botle_neck_train = model.predict_generator(
        generator=train_generator,
        steps= number_train//batch_size,
    )
    botle_neck_validate = model.predict_generator(
        generator=validate_generator,
        steps=number_validate // batch_size,
    )

    np.save(open(file_output_pre_train,'wb'),botle_neck_train)
    np.save(open(file_output_pre_validate,'wb'),botle_neck_validate)

def train():
    input_train = np.load(open(file_output_pre_train,'rb'))
    input_validate = np.load(open(file_output_pre_validate,'rb'))
    y_train = np.concatenate((np.array([0]*int(number_train/2)),np.array([1]*int(number_train/2))))
    y_validate = np.concatenate((np.array([0]*int(number_validate/2)),np.array([1]*int(number_validate/2))))


    top_model = Sequential()
    top_model.add(Flatten(input_shape=input_train.shape[1:]))
    top_model.add(Dense(
        units=50,
        activation= relu,
    ))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(
        units=1,
        activation= sigmoid,
    ))

    top_model.compile(
        optimizer= 'rmsprop',
        loss=binary_crossentropy,
        metrics=['accuracy'],
    )
    top_model.fit(
        x = input_train,
        y = y_train,
        batch_size= batch_size,
        epochs= epochs,
        validation_data=(input_validate,y_validate),
    )
    # top_model.save_weights(filepath=file_save_weight)
    top_model.load_weights(filepath=file_save_weight)

create_input()
train()