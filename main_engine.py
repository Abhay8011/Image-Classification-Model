import numpy as np
from dataset import OperationDataset
from model import load_model
from keras.optimizers import Adamax
from keras.callbacks import ModelCheckpoint
from Image_performance_visualization import Visualization
from sklearn.utils import shuffle

print("============== Starting to Load the dataset==============")

dataThing = OperationDataset()

train_path = input("Please enter the path for the training data: ")
training_img, training_label = dataThing.load_dataset(train_path)

label_to_id = {v:k for k,v in enumerate(np.unique(training_label))}
id_to_label = {v:k for k,v in label_to_id.items()}

print("~> Following are the label names with their respective id: \n")
print(id_to_label)

training_label_id = np.array([label_to_id[i] for i in training_label])
print("~> Training Label Id: \n")
print(training_label_id)

print(f'~> Shape for training data set: {training_img.shape} \n')
print(f'~> Shape for training label set: {training_label_id.shape} \n')

test_path = input("Please enter the path for the test data: ")
validation_img, validation_label = dataThing.load_dataset(test_path)

print("============== Dataset loading complete !! ==============")

validation_label_id = np.array([label_to_id[i] for i in validation_label])

print(f'~> Shape for test data set: {validation_img.shape} \n')
print(f'~> Shape for test label set: {validation_label_id.shape} \n')

print("============== Preprocessing Starts ==============")
x_train, y_train, x_test, y_test = dataThing.data_preprocess(training_img, validation_img, training_label_id,
                                                             validation_label_id)

x_valid, x_test = x_test[:1000], x_test[1000:]
y_valid, y_test = y_test[:1000], y_test[1000:]

visualize = Visualization()

visualize.image_visualize(x_train)

print("============== Preprocessing complete !! ==============")

print("============== Preparation starts for modelling ==============")

print("-----------------1. Loading the model-------------------------")
model = load_model()

print("~> Here is the overall model summary: \n")
model.summary()

print("----------------2. Compiling the model----------------------")
model.compile(loss= 'categorical_crossentropy',
              optimizer= Adamax(),
              metrics=['accuracy'])

print("Model Compiled")


checkpointer = ModelCheckpoint(filepath = 'cnn_from_scratch_fruits.hdf5', verbose = 1, save_best_only = True)

print("-----------------3. Training Starts Here--------------------")

history = model.fit(x_train,y_train,
                    batch_size = 32,
                    epochs=30,
                    validation_data=(x_valid, y_valid),
                    callbacks = [checkpointer],
                    verbose=2, shuffle=True)

print("=================== Training Completed ====================")

## Saving the Model
model.save("model_ibm_image_classification.h5")

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

y_pred = model.predict(x_test)

visualize.test_pred_visualize(x_test,y_test,y_pred,training_label)

visualize.accuracy_visualize(history)







