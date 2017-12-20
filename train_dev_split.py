import glob
import os

from sklearn.model_selection import train_test_split


def one_random_split(dev_size, test_size):
    """
    Author: Trinh Man Hoang
    :param  dev_size : size of dev_set < 1
    :return:
        X_train, X_dev, Y_train, Y_dev respectively
    :usage:
        from . import train_dev_split as spl
        X, Y = spl.one_random_split(devsize)
    """

    i_l_ROOT = 'images'
    f = os.listdir('images')

    X = []
    Y = []

    for folder in f:
        direct = os.path.join(i_l_ROOT,folder)
        l_direct = os.path.join(direct, folder + ".txt")


        with open(l_direct) as label_file:
            label_list = label_file.read().splitlines()

        file_list = glob.glob(os.path.join(direct, '*.jpg'))
        for i in range(len(file_list)):
            X.append(file_list[i])
            Y.append(label_list[i])

    X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=dev_size)
    X_dev, X_test,Y_dev,Y_test = train_test_split(X_dev, Y_dev, test_size= test_size)

    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test


def k_split_sample(k_set, dev_size, test_size):
    """
    Author: Trinh Man Hoang
    :param  k_set: #train_split sets
            dev_size : size of dev_set < 1
    :return:
        k_set folders (in db_Root) contains random splitting
    :usage:
        from . import train_dev_split as spl
        spl.k_split_sample(kset, devsize)
    """


    db_ROOT = 'db'
    db_ROOT = os.path.abspath(db_ROOT)

    for k in range(k_set):

        db_direct = os.path.join(db_ROOT,"db" + str(k))
        if not os.path.exists(db_direct):
            os.makedirs(db_direct)
        else:
            trash_files = os.listdir(db_direct)
            for trash in trash_files:
                os.remove(db_direct + trash)

        train_file = open(os.path.join(db_direct, "train.txt"), "w+")
        test_file = open(os.path.join(db_direct,"test.txt"), "w+")
        dev_file = open(os.path.join(db_direct, "dev.txt"), "w+")
        lbtrain_file = open(os.path.join(db_direct,"lbtrain.txt"), "w+")
        lbtest_file = open(os.path.join(db_direct, "lbtest.txt"), "w+")
        lbdev_file = open(os.path.join(db_direct,"lbdev.txt"), "w+")

        X_train, X_dev, X_test, Y_train, Y_dev, Y_test = one_random_split(dev_size= dev_size, test_size= test_size)

        for i in range(len(X_train)):
             train_file.write(X_train[i] + "\n")
             lbtrain_file.write(Y_train[i] + "\n")

        for i in range(len(X_dev)):
             dev_file.write(X_dev[i] + "\n")
             lbdev_file.write(Y_dev[i] + "\n")

        for i in range(len(X_test)):
            test_file.write(X_test[i] + "\n")
            lbtest_file.write(Y_test[i] + "\n")

        train_file.close()
        test_file.close()
        dev_file.close()
        lbtrain_file.close()
        lbtest_file.close()
        lbdev_file.close()

#k_split_sample(5, 0.4, 0.5)
