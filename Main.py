from Model import ResNet50
import support_function
import os
from keras.preprocessing.image import ImageDataGenerator

Root = 'db'
folders = os.listdir(Root)
acc = 0

for folder in folders:

    direct = os.path.join(Root, folder)
    model = ResNet50(input_shape= (150,150, 3), classes=15)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = support_function.load_data(direct)

    train_generator = train_datagen.flow(
        X_train,
        Y_train)

    model.fit_generator(train_generator, steps_per_epoch=32, epochs=5)

    # model.fit(X_train, Y_train, epochs=5, batch_size=16)
    model.fit(X_dev, Y_dev, epochs=5, batch_size=16)

    preds = model.evaluate(X_test, Y_test)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))
    acc += preds[1]

model.save('my_ResNet50.h5')
print('Mean Accuracy: %0.2f'%(acc/len(folders)))

