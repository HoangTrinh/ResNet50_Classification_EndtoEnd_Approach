from Model import ResNet50
import support_function
import os
from keras.preprocessing.image import ImageDataGenerator

"""
Author: Trinh Man Hoang _ update: 20/12/2017
    Usage:
    1. Run label_marker.mark_label()
    2. Run train_dev_split.k_split_sample()
    3. Run support_function.save_data()
    4. Run this file.
    
    => vegetable classification by ResNet50 - hand coding
"""



Root = 'db'
folders = os.listdir(Root)
acc = 0

for folder in folders:

    direct = os.path.join(Root, folder)

    # Create model
    model = ResNet50(input_shape= (150,150, 3), classes=15)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train_data generator ( data augmentation)
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # load data
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = support_function.load_data(direct)

    # fit train data to generator
    train_generator = train_datagen.flow(
        X_train,
        Y_train)

    # train generated data
    model.fit_generator(train_generator, steps_per_epoch=32, epochs=5)

    # train a part of original data
    model.fit(X_dev, Y_dev, epochs=5, batch_size=16)

    # prediction
    preds = model.evaluate(X_test, Y_test)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))
    acc += preds[1]

model.save('my_ResNet50.h5')
print('Mean Accuracy: %0.2f'%(acc/len(folders)))

