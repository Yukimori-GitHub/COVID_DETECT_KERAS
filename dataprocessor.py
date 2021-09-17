import os

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt, image as mpimg


def data_process():
    main_dir = "D:\\dataset\\datasets\\Data"
    train_dir = os.path.join(main_dir, 'train')
    test_dir = os.path.join(main_dir, 'test')

    train_covid_dir = os.path.join(train_dir, 'COVID19')
    train_normal_dir = os.path.join(train_dir, 'NORMAL')
    test_covid_dir = os.path.join(test_dir, 'COVID19')
    test_normal_dir = os.path.join(test_dir, 'NORMAL')

    train_covid_names = os.listdir(train_covid_dir)
    # print(train_covid_names[:10])

    train_normal_names = os.listdir(train_normal_dir)
    # print(train_normal_names[:10])

    test_covid_names = os.listdir(test_covid_dir)
    # print(test_covid_names[:10])

    test_normal_names = os.listdir(test_normal_dir)
    # print(test_normal_names[:10])

    # print("Total images present in the training set: ", len(train_covid_names + train_normal_names))
    # print("Total images present in the testing set: ", len(test_covid_names + test_normal_names))

    rows = 4
    cols = 4
    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    covid_pic = [os.path.join(train_covid_dir, filename) for filename in train_covid_names[0:8]]
    normal_pic = [os.path.join(train_normal_dir, filename) for filename in train_normal_names[0:8]]
    # print(covid_pic)
    # print(normal_pic)
    merged_list = covid_pic + normal_pic
    for i, img_path in enumerate(merged_list):
        data = img_path.split('\\', 6)[6]
        sp = plt.subplot(rows, cols, i + 1)
        sp.axis('Off')
        img = mpimg.imread(img_path)
        sp.set_title(data, fontsize=10)
        plt.imshow(img, cmap='gray')

    # plt.show()

    dgen_train = ImageDataGenerator(rescale=1.0 / 255,
                                    validation_split=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
    dgen_validation = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    dgen_test = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = dgen_train.flow_from_directory(train_dir,
                                                     target_size=(150, 150),
                                                     subset='training',
                                                     batch_size=32,
                                                     class_mode='binary')

    validation_generator = dgen_train.flow_from_directory(train_dir,
                                                          target_size=(150, 150),
                                                          subset='validation',
                                                          batch_size=32,
                                                          class_mode='binary')

    test_generator = dgen_test.flow_from_directory(test_dir,
                                                   target_size=(150, 150),
                                                   batch_size=32,
                                                   class_mode='binary')
    train_generator.class_indices
    train_generator.image_shape

    return train_generator, validation_generator, test_generator
