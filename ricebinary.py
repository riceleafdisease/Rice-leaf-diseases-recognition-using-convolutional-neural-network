import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import numpy as np
import cv2
import os
from os import listdir
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc,roc_auc_score,accuracy_score,precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
from numpy import expand_dims
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc,roc_auc_score
import glob
import tensorflow as tf 
tf.compat.v1.reset_default_graph
img_rows, img_cols = 256, 256
seed_value= 0
np.random.seed(seed_value)

# Read train images from train folder 
 
data=[]
label_list = []
for filename in glob.glob('Ricedatasetbinary/dataset1/train/*/*.jpg'):
    #print(os.path.basename(filename))
    only_file_name = os.path.basename(filename)## hispa_2_output.jpg
    train_img_label = only_file_name.split('_')[0]
    img = load_img(filename)
    img = img_to_array(img)
    data.append(img)
    label_list.append(train_img_label)
    
np_data = np.array(data, dtype=np.float16) / 255.0
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)

# Read validation images from validation folder 
    
data1=[]
label_list1 = []
for filename in glob.glob('Ricedatasetbinary/dataset1/validation/*/*.jpg'):
   # print(os.path.basename(filename))
    only_file_name1 = os.path.basename(filename)## hispa_2_output.jpg
    train_img_label1 = only_file_name1.split('_')[0]
    img1 = load_img(filename)
    img1 = img_to_array(img1)
    data1.append(img1)
    label_list1.append(train_img_label1)
    
np_data1 = np.array(data1, dtype=np.float16) / 255.0
label_binarizer = LabelBinarizer()
image_labels1 = label_binarizer.fit_transform(label_list1)

# Read test images from test folder 

data2=[]
label_list2 = []
for filename in glob.glob('Ricedatasetbinary/dataset1/test/*/*.jpg'):
    #print(os.path.basename(filename))
    only_file_name2 = os.path.basename(filename)## hispa_2_output.jpg
    train_img_label2 = only_file_name2.split('_')[0]
    img2 = load_img(filename)
    img2 = img_to_array(img2)
    data2.append(img2)
    label_list2.append(train_img_label2)
    
np_data2 = np.array(data2, dtype=np.float16) / 255.0
label_binarizer = LabelBinarizer()
image_labels2 = label_binarizer.fit_transform(label_list2)


classifier = Sequential()


# step - 1 - Convolution
classifier.add(Convolution2D(16, (3, 3), input_shape=(img_rows, img_cols, 3), padding='valid'))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(32, (3, 3), padding='valid'))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(64, (3, 3), padding='valid'))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())
classifier.add(Dense(64))
classifier.add(Activation('relu'))
classifier.add(Dropout(0.05))
classifier.add(Dense(2))
classifier.add(Activation('softmax'))
# Compiling the CNN
optimizer = Adam(lr=0.001,beta_1=0.9, beta_2=0.999)

classifier.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)


hist=classifier.fit( np_data,image_labels,batch_size=32,
    verbose=1,
    nb_epoch=50,
    validation_data=(np_data1,image_labels1)

)

#confusion_matrics
Y_pred = classifier.predict(np_data2)
y_pred = np.argmax(Y_pred, axis=1)

target_names = [ 'Blast', 'Non Blast']
					
print(classification_report(np.argmax(image_labels2,axis=1), y_pred,target_names=target_names))

print(confusion_matrix(np.argmax(image_labels2,axis=1), y_pred))
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlim(-0.5, 1.5)
    plt.ylim(1.5, -0.5)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(image_labels2,axis=1), y_pred))

np.set_printoptions(precision=2)

plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')

plt.show()


# visualizing losses and accuracy

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
xc=range(50)
plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
#
#
#
## summarize feature map shapes
for i in range(len(classifier.layers)):
	layer = classifier.layers[i]
	print(i, layer.name, layer.output.shape)
filters, biases = classifier.layers[6].get_weights()
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
n_filters, ix = 6, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	for j in range(3):
		# specify subplot and turn of axis
		ax = plt.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		plt.imshow(f[:, :, j], cmap='gray')
		ix += 1

## show the figure
plt.show()
#
##load the image with the required shape
img = load_img('Ricedatasetbinary/dataset1/test/Blast/Blast_0025.jpg', target_size=(256, 256))
# convert the image to an array
img1 = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img1 = expand_dims(img1, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img1 /= 255.
plt.show()
print(img1.shape)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes1 = classifier.predict_classes(images, batch_size=10)
imgplot = plt.imshow(img)
plt.title('Blast')
plt.show()
print(classes1)
if classes1==0:
    print('Predicted class is: Blast')
else:
    print('Predicted class is: Non Blast')
     
print("Predicted class is:",classes1)
#
#
from keras.models import Model
layer_outputs = [layer.output for layer in classifier.layers]
activation_model = Model(inputs=classifier.input, outputs=layer_outputs)
activations = activation_model.predict(img1)
 
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2,col_size*2))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1
display_activation(activations, 3, 3, 3)

# ROC Curve

from scipy import interp
from itertools import cycle
n_classes=2
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(image_labels2[:, i],Y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(image_labels2.ravel(), Y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
lw = 2
plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 14})
plt.grid()
plt.plot(fpr[0], tpr[0], color='darkorange',
lw=lw, label='ROC curve of Blast (area = %0.2f)' % roc_auc[0])
plt.plot(fpr[1], tpr[1], color='aqua',
lw=lw, label='ROC curve of Non Blast (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
code_binary.py
Displaying code_binary.py.