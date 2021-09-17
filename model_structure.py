from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from dataprocessor import data_process
import os


def build_model(model):
    model.add(Conv2D(32, (5, 5), padding='SAME', activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (5, 5), padding='SAME', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model


def run(model):
    model.compile(Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


def evaluator(model):
    (train_generator, validation_generator, test_generator) = data_process()
    history = model.fit(train_generator,
                        batch_size=32,
                        epochs=30,
                        validation_data=validation_generator,
                        verbose=1)
    print(history.history.keys())
    model.save('D:\\Program Files\\pythonProject\\covid.h5')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.title('Training and Validation losses')
    plt.xlabel('epoch')

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['Training', 'Validation'])
    plt.title('Training and Validation accuracy')
    plt.xlabel('epoch')

    test_loss, test_acc = model.evaluate(test_generator)
    print('test loss: {} test acc: {}'.format(test_loss, test_acc))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = Sequential()
    model = build_model(model)
    run(model)
    evaluator(model)
    # data_process()

