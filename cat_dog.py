from keras.layers import *
from keras.models import *
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

with h5py.File('fv_Xception.h5', 'r') as h:
    x_train = np.array(h['train'])
    x_test = np.array(h['test'])
    y_train = np.array(h['label'])

idx = np.arange(x_train.shape[0])
np.random.shuffle(idx)
x_train = x_train[idx]
y_train = y_train[idx]

x = Input(shape=[x_train.shape[1]])
y = Dropout(0.5)(x)
y = Dense(1, activation='sigmoid', name='classifier')(y)

model = Model(x, y, name='GAP')

model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=20, epochs=5, validation_split=0.2, verbose=2)

y_pred = model.predict(x_test, batch_size=20, verbose=1)
y_pred = y_pred.clip(min=0.005, max=0.995)

table = pd.read_csv('sample_submission.csv')
gen = ImageDataGenerator()
test_gen = gen.flow_from_directory('test_processed', [229, 229], shuffle=False,
                                   batch_size=20, class_mode=None)

for i, filename in enumerate(test_gen.filenames):
    index = int(filename[filename.rfind('\\') + 1:filename.rfind('.')])
    table.set_value(index - 1, 'label', y_pred[i])

table.to_csv('pred.csv', index=None)
