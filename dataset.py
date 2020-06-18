import numpy as np
import os
import cv2
import keras
import pandas as pd
import glob

class OperationDataset():

    def __init__(self):
        pass

    def load_dataset(self,path):

        training_img = []
        training_label = []
        path = str(path)
        for dir_path in glob.glob(path):
            img_label = dir_path.split("/")[-1]
            for image_path in glob.glob(os.path.join(dir_path,"*.jpg")):
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (128, 128))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                training_img.append(image)
                training_label.append(img_label)
        training_img = np.array(training_img)
        training_label = np.array(training_label)
        return training_img, training_label

    def data_preprocess(self,training_img, validation_img, training_label_id, validation_label_id):
        x_train, x_test = training_img, validation_img
        y_train, y_test = training_label_id, validation_label_id

        x_train = x_train/255
        x_test = x_test/255

        y_train = keras.utils.to_categorical(y_train,6)
        y_test = keras.utils.to_categorical(y_test,6)

        print('~> Original Sizes: \n')
        print(f'Train Dataset: {x_train.shape} & {y_train.shape} \n')
        print(f'Test Dataset: {x_test.shape} & {y_test.shape} \n')

        return x_train, y_train, x_test, y_test
