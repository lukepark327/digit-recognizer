import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping


random_seed = 42
np.random.seed(random_seed)

leakyrelu = LeakyReLU(alpha=0.3)


# CNN model
def create_model(result_class_size):
    model = Sequential()

    # use Conv2D to create our first convolutional layer, with 32 filters, 5x5 filter size,
    # input_shape = input image with (height, width, channels), activate ReLU to turn negative to zero
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    # add a pooling layer for down sampling
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # add another conv layer with 32 filters, 3x3 filter size,
    model.add(Conv2D(32, (3, 3), activation='relu'))
    # add another conv layer with 16 filters, 3x3 filter size,
    model.add(Conv2D(16, (3, 3), activation='relu'))

    # set 20% of the layer's activation to zero, to void overfit
    model.add(Dropout(0.2))
    # convert a 2D matrix in a vector
    model.add(Flatten())

    # add fully-connected layers, and ReLU activation
    model.add(Dense(130, activation=leakyrelu))
    model.add(Dense(50, activation=leakyrelu))

    # add a fully-connected layer with softmax function to squash values to 0...1
    model.add(Dense(result_class_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    return model


if __name__ == "__main__":
    # load training and testing csv files
    df_train = pd.read_csv('./train.csv')
    df_test = pd.read_csv('./test.csv')

    # input X: every columns but the first
    df_train_x = df_train.iloc[:, 1:]
    # output Y: only the first column
    df_train_y = df_train.iloc[:, :1]

    # turn the label to 42000 binary class matrix
    arr_train_y = np_utils.to_categorical(df_train_y['label'].values)

    # define the model output size and get the summary
    model = create_model(arr_train_y.shape[1])
    model.summary()

    # normalize 255 grey scale to values between 0 and 1
    df_test = df_test / 255.0
    df_train_x = df_train_x / 255.0

    # reshape training X and text x to (number, height, width, channels)
    arr_train_x_28x28 = np.reshape(df_train_x.values, (df_train_x.values.shape[0], 28, 28, 1))
    arr_test_x_28x28 = np.reshape(df_test.values, (df_test.values.shape[0], 28, 28, 1))

    # validate size = 8%
    split_train_x, split_val_x, split_train_y, split_val_y, = train_test_split(
        arr_train_x_28x28, arr_train_y, test_size=0.08, random_state=random_seed)

    # define model callback
    reduce_lr = ReduceLROnPlateau(monitor='val_acc',
                                  factor=0.5,
                                  patience=3,
                                  min_lr=0.00001)

    # define image generator
    datagen = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally
        height_shift_range=0.1  # randomly shift images vertically
    )
    datagen.fit(split_train_x)

    # train the model with callback and image generator
    epochs = 30
    batch_size = 64

    earlystopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)

    model.fit_generator(datagen.flow(split_train_x, split_train_y, batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(split_val_x, split_val_y),
                        verbose=2,
                        callbacks=[reduce_lr, earlystopping])

    # predict the result and save it as a csv file
    prediction = model.predict_classes(arr_test_x_28x28, verbose=0)
    data_to_submit = pd.DataFrame({"ImageId": list(range(1, len(prediction) + 1)), "Label": prediction})
    data_to_submit.to_csv("result.csv", header=True, index=False)
