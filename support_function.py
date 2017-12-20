from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras import backend as K
import os

def save_data(db_direct, datatype):
    """
    Author: Trinh Man Hoang
    :param  db_direct : direction of data folder
            datatype : train, dev, test
    :return:
        two .npy files contain encoded data and encoded labels at db_direct correlate to datatype
    :usage:
        from . import support_function as sp
        sp.save_data()
    """


    img_width, img_height = 150, 150
    X = []

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    with open(os.path.join(db_direct, str(datatype) + '.txt')) as name_file:
        names = name_file.read().splitlines()
    for name in names:
        img = load_img(name)
        x = img_to_array(img)
        x = np.resize(x, input_shape)
        X.append(x)
    X = np.array(X,ndmin=4)
    X = X/255

    with open(os.path.join(db_direct, 'lb'+ str(datatype) + '.txt')) as label_file:
        Y = label_file.read().splitlines()
    Y = list(map(int,Y))
    Y = np.array(Y)
    Y = Y.reshape((1, Y.shape[0]))
    Y = np.eye(15)[Y.reshape(-1)]

    np.save(os.path.join(db_direct, datatype +'_image_data.npy'),X)
    np.save(os.path.join(db_direct, datatype + '_image_label_one_hot.npy'),Y)


def load_data(db_direct):
    """
        Author: Trinh Man Hoang
        :param  db_direct : direction of data folder
        :return:
            Six numpy array contain encoded data and encoded labels at db_direct correlate to datatype
        :usage:
            from . import support_function as sp
            X,Y, ... = sp.load_data()
    """


    X_train = np.load(os.path.join(db_direct,'train_image_data.npy'))
    Y_train = np.load(os.path.join(db_direct,'train_image_label_one_hot.npy'))
    X_dev = np.load(os.path.join(db_direct, 'dev_image_data.npy'))
    Y_dev = np.load(os.path.join(db_direct,'dev_image_label_one_hot.npy'))
    X_test = np.load(os.path.join(db_direct,'test_image_data.npy'))
    Y_test = np.load(os.path.join(db_direct, 'test_image_label_one_hot.npy'))
    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test



# db_Root = 'db'
# db_folders = os.listdir(db_Root)
# for folder in db_folders:
#     save_data(os.path.join(db_Root, folder), 'train')
#     save_data(os.path.join(db_Root, folder), 'dev')
#     save_data(os.path.join(db_Root, folder), 'test')