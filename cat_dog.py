from keras.layers import *
from keras.models import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import pandas as pd

from sklearn.svm import SVC

import re


# load feature vectors from .h5 files
def load_fv(fv_files):
    fv_train, fv_test, label_train = ([], [], [])
    for fv in fv_files:
        with h5py.File(fv, 'r') as h5:
            fv_train.append(np.array(h5['train']))
            fv_test.append(np.array(h5['test']))
            label_train = np.array(h5['label'])
    fv_train = np.concatenate(fv_train, axis=1)
    fv_test = np.concatenate(fv_test, axis=1)

    return fv_train, fv_test, label_train


# write prediction into .csv file
def write_pred(pred, template, dst, clip=False):
    if clip:
        pred = pred.clip(min=0.004, max=0.996)
    df = pd.read_csv(template)
    gen = ImageDataGenerator()
    img_size = [299, 299]
    test_gen = gen.flow_from_directory('test_processed', img_size, shuffle=False,
                                       batch_size=16, class_mode=None)
    # write the df in the correct order
    for i, filename in enumerate(test_gen.filenames):
        index = int(filename[filename.rfind('\\') + 1:filename.rfind('.')])
        df.set_value(index - 1, 'label', pred[i])
    # save df
    df.to_csv(dst, index=None)


def train(batch_size, epochs, early_stop=True, patience=3, method='FC'):
    if method not in ['FC', 'SVM']:
        raise ValueError('Arg method must be either \'FC\' or \'SVM\'.')

    # h5_files = ['fv_Xception.h5', 'fv_InceptionV3.h5', 'fv_InceptionResNetV2.h5']
    h5_files = ['fv_Xception.h5', 'fv_InceptionV3.h5', 'fv_InceptionResNetV2.h5']
    x_train, x_test, y_train = load_fv(h5_files)

    # shuffle feature vectors
    idx = np.arange(x_train.shape[0])
    np.random.shuffle(idx)
    x_train = x_train[idx]
    y_train = y_train[idx]

    # build model
    x = Input(shape=[x_train.shape[1]])
    y = Dense(2048, activation='relu')(x)
    y = Dropout(0.5)(y)
    y = Dense(1024, activation='relu')(y)
    if method == 'SVM':
        y = Dense(20, activation='sigmoid', name='svm')(y)
    y = Dense(1, activation='sigmoid', name='classifier')(y)

    model = Model(x, y, name='GAP')

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    if early_stop:
        es = EarlyStopping(monitor='val_loss', patience=patience)
        callbacks = [es]
    else:
        callbacks = None
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2,
              callbacks=callbacks, verbose=2)

    if method == 'SVM':
        svm_model = Model(inputs=model.input, outputs=model.get_layer(name='svm').output)
        svm_train = svm_model.predict(x_train, batch_size=16, verbose=1)

        clf = SVC()
        clf.fit(svm_train, y_train)

        svm_test = svm_model.predict(x_test, batch_size=16, verbose=1)

        # predict test set
        y_pred = clf.predict(svm_train)
        y_pred = np.clip(y_pred, a_min=0.005, a_max=0.995)
        log_loss = -np.sum(y_train * np.log(y_pred) + (1 - y_train) * np.log(1 - y_pred)) / len(
            y_train)

        print('\nlogloss\n', log_loss)

        y_pred = clf.predict(svm_test)
    else:
        y_pred = model.predict(x_test)

    write_pred(y_pred, 'sample_submission.csv', 'pred.csv', clip=True)

    # predict train set
    # x_train, _, _ = load_fv(h5_files)
    #
    # y_pred = model.predict(x_train, batch_size=16, verbose=1)
    # df = pd.read_csv('train_pred.csv')
    # train_gen = gen.flow_from_directory('train_classified', img_size, shuffle=False,
    #                                     batch_size=16)
    #
    # idx_patten = re.compile(r'\.(\d+)\.')
    # for i, filename in enumerate(train_gen.filenames):
    #     kind = filename[0:3]
    #     flip = filename.__contains__('flip')
    #     index = int(idx_patten.search(filename).group(1))
    #     if kind == 'dog':
    #         index += 25000
    #     if flip:
    #         index += 12500
    #     df.set_value(index - 1, 'label', y_pred[i])
    #
    # df.to_csv('train_pred.csv', index=None)


def avr_model(epochs):
    h5_files = ['fv_Xception.h5']
    x_train, x_test, y_train = load_fv(h5_files)
    # shuffle feature vectors
    idx = np.arange(x_train.shape[0])
    np.random.shuffle(idx)
    x_train = x_train[idx]
    y_train = y_train[idx]
    # build model
    x = Input(shape=[x_train.shape[1]])
    y = Dense(1024, activation='relu')(x)
    y = Dropout(0.5)(y)
    y = Dense(512, activation='relu')(y)
    y = Dense(1, activation='sigmoid', name='classifier')(y)
    model = Model(x, y, name='Xception_GAP')
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=epochs[0], validation_split=0.2,
              callbacks=None, verbose=2)
    xception_pred = model.predict(x_test)

    h5_files = ['fv_InceptionV3.h5']
    x_train, x_test, y_train = load_fv(h5_files)
    # shuffle feature vectors
    x_train = x_train[idx]
    y_train = y_train[idx]
    # build model
    x = Input(shape=[x_train.shape[1]])
    y = Dense(1024, activation='relu')(x)
    y = Dropout(0.5)(y)
    y = Dense(512, activation='relu')(y)
    y = Dense(1, activation='sigmoid', name='classifier')(y)
    model = Model(x, y, name='IceptionV3_GAP')
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=epochs[1], validation_split=0.2,
              callbacks=None, verbose=2)
    inception_pred = model.predict(x_test)

    h5_files = ['fv_InceptionResNetV2.h5']
    x_train, x_test, y_train = load_fv(h5_files)
    # shuffle feature vectors
    x_train = x_train[idx]
    y_train = y_train[idx]
    # build model
    x = Input(shape=[x_train.shape[1]])
    y = Dense(1024, activation='relu')(x)
    y = Dropout(0.5)(y)
    y = Dense(512, activation='relu')(y)
    y = Dense(1, activation='sigmoid', name='classifier')(y)
    model = Model(x, y, name='IceptionResNetV2_GAP')
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=epochs[2], validation_split=0.2,
              callbacks=None, verbose=2)
    inceptionres_pred = model.predict(x_test)

    acc = [0.790, 0.788, 0.804]

    y_pred = (xception_pred * acc[0] + inception_pred * acc[1] + inceptionres_pred * acc[2]) / np.sum(acc)

    write_pred(y_pred, 'sample_submission.csv', 'pred.csv', clip=True)


# train(batch_size=128, epochs=20, early_stop=False, method='FC')

avr_model(epochs=[12, 20, 12])
