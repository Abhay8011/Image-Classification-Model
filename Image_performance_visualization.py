import matplotlib.pyplot as plt
import random
import numpy as np

class Visualization():
    def __init__(self):
        pass

    def image_visualize(self,x_train):
        fig = plt.figure(figsize=(30,5))
        for i,j in enumerate(random.sample(range(1, x_train.shape[0]),10)):
            ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
            ax.imshow(np.squeeze(x_train[j]))

    def test_pred_visualize(self, x_test, y_test, y_pred, target_labels):
        fig = plt.figure(figsize=(16,9))
        for i,idx in enumerate(np.random.choice(x_test.shape[0],size=16,replace=False)):
            ax=fig.add_subplot(4,4,i+1,xticks=[],yticks=[])
            ax.imshow(np.squeeze(x_test[idx]))
            pred_idx = np.argmax(y_pred[idx])
            true_idx = np.argmax(y_test[idx])
            ax.set_title("{} ({})".format(target_labels[pred_idx],target_labels[true_idx]),
                         color=("green" if pred_idx==true_idx else "red"))

    def accuracy_visualize(self,history):
        plt.figure(1)

        plt.subplot(211)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        plt.subplot(212)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
