from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

import h5py


def write_fv(model_func, img_size, preprocessor):
    width, height = img_size
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if preprocessor:
        x = Lambda(preprocessor)(x)

    base_model = model_func(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    gen = ImageDataGenerator()
    train_gen = gen.flow_from_directory('train_classified', img_size, shuffle=False,
                                        batch_size=20)
    test_gen = gen.flow_from_directory('test_processed', img_size, shuffle=False,
                                       batch_size=20, class_mode=None)

    train = model.predict_generator(train_gen, train_gen.samples/train_gen.batch_size)
    test = model.predict_generator(test_gen, test_gen.samples/test_gen.batch_size)
    with h5py.File('fv_{0}.h5'.format(model_func.__name__)) as h:
        h.create_dataset('train', data=train)
        h.create_dataset('test', data=test)
        h.create_dataset('label', data=train_gen.classes)


write_fv(Xception, [299, 299], xception.preprocess_input)
