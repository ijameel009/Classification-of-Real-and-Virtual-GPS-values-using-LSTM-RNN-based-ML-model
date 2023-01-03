#Load Packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


from dataloader import LoadGPSData

#%%
# Seed value (can actually be different for each attribution step)
seed_value = 6

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value) # tensorflow 2.x
# tf.set_random_seed(seed_value) # tensorflow 1.x

#%%

sequence_length = 4
epochs = 100
batch_size = 64

gps_data_loader = LoadGPSData(sequence_length = sequence_length)

X, y = gps_data_loader.get_data("Dataset")


model = Sequential()
model.add(LSTM(16,input_shape=(sequence_length,3),return_sequences=False)) #True = many to many
model.add(Dense(16, activation='linear'))
model.add(Dense(64, activation='linear'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='bce',optimizer ='adam',metrics=['accuracy'])
print(model.summary())


#%%
#training loss and Validation Loss Plots
history = model.fit(X,y,epochs=epochs,batch_size=batch_size,validation_split=0.2);
model.save('RealClassifier.h5')

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%%
#Training Accuracy and Validation Accuracy
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
plt.plot(epochs, accuracy, 'y', label='Training acc')
plt.plot(epochs, val_accuracy, 'r', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
scores = model.evaluate(X,y,batch_size=batch_size)
print('Accurracy: {}'.format(scores[1]*100.0),"%")

#How do we know how it is doing for real GPS and Virtual GPS
#Confusion matrix
#We compare labels and plot them based on correct or wrong predictions.
#Since sigmoid outputs probabilities
#%%
mythreshold = 0.694
y_pred = (model.predict(X)>= mythreshold).astype(int)
cm = confusion_matrix(y, y_pred)
print(cm)


#Need to balance positive, negative, false positive and false negative.
#ROC can help identify the right threshold.

#%%
from sklearn.metrics import roc_curve
y_preds = model.predict(X).ravel()
fpr, tpr, thresholds = roc_curve(y, y_preds)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'y--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()

#Receiver Operating Characteristic (ROC) Curve is a plot that helps us
#visualize the performance of a binary classifier when the threshold is varied.
#One way to find the best threshold once we calculate the true positive
#and false positive rates is ...
#The optimal cut off point would be where “true positive rate” is high
#and the “false positive rate” is low.
#Based on this logic let us find the threshold where tpr-(1-fpr) is zero (or close to 0)
#%%
import pandas as pd
i = np.arange(len(tpr))
roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'thresholds' : pd.Series(thresholds, index=i)})
ideal_roc_thresh = roc.iloc[(roc.tf-0).abs().argsort()[:1]]  #Locate the point where the value is close to 0
print("Ideal threshold is: ", ideal_roc_thresh['thresholds'])
#Now use this threshold value in the confusion matrix to visualize the balance
#between tp, fp, fp, and fn

#%%
#Area under the curve (AUC) for ROC plot can be used to understand hpw well a classifier
#is performing.
#% chance that the model can distinguish between positive and negative classes.

from sklearn.metrics import auc
auc_value = auc(fpr, tpr)
print("Area under curve, AUC = ", auc_value)

#%%
#defining the location of test data and predicting the values for each sequence
#value above threshold will be 1 for virtual value and below threshold will be labeled as 0.
X_hat = gps_data_loader.get_test_data("E:\Masters Semester 3\Machine Learning\Project\LSTM_Model\LSTM_Model\Dataset\data test")

print("Predicted values are: \n", model.predict(X_hat))



