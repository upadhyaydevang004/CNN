import keras
from keras.models import Sequential
from mlxtend.data import loadlocal_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from numpy import array, argmax
import matplotlib.pyplot as plt
import pandas

#load mnist dataset
# (X_train, y_train), (X_test, y_test) = mnist.load_data() #everytime loading data won't be so easy :)
def load_mnist():
	X_train, y_train = loadlocal_mnist(images_path='train-images.idx3-ubyte',labels_path='train-labels.idx1-ubyte')
	X_test, y_test =  loadlocal_mnist(images_path='t10k-images.idx3-ubyte',labels_path='t10k-labels.idx1-ubyte')

	img_rows, img_cols = 28, 28

	#reshaping
	if K.image_data_format() == 'channels_first':
	    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
	    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
	    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
	    input_shape = (img_rows, img_cols, 1)
	#more reshaping
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	print('X_train shape:', X_train.shape) #X_train shape: (60000, 28, 28, 1)
	
	return X_train,y_train,X_test,y_test, input_shape


def Output_CSV(prediction):
	values = prediction
	values = array(values)

	s = pandas.Series(values)
	predict = pandas.get_dummies(s)
	# print(predict)
	predict = predict.astype(dtype = 'int')	
	pandas.DataFrame(predict).to_csv('mnist.csv',header = False, index = False)

def plot(model_history,lr):
	model_history = model_history[0]
	plt.subplots_adjust(hspace=0.5)
	plt.subplot(2, 1, 1)
	plt.title("Loss (Eta = %.4f)" % lr)
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	line1, = plt.plot(model_history['loss'], label='Train Loss')
	line2, = plt.plot(model_history['val_loss'], label='Validation Loss')
	plt.legend(handles=[line1, line2])
	plt.grid()

	plt.subplot(2, 1, 2)
	plt.title("Accuracy (Eta = %.4f)" % lr)
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	line1, = plt.plot(model_history['acc'], label='Train Accuracy')
	line2, = plt.plot(model_history['val_acc'], label='Validation Accuracy')
	plt.legend(handles=[line1, line2])
	plt.grid()
	plt.show()


def create_model(input_shape,num_category):
	model = None
	model = Sequential()
	#convolutional layer with rectified linear unit activation
	model.add(Conv2D(10, kernel_size=(5, 5),
	                 activation='relu',
	                 input_shape=input_shape,padding="valid"))
	#10 convolution filters used each of size 5x5
	#again
	model.add(Conv2D(20, (5, 5), activation='relu',padding="valid"))
	#20 convolution filters used each of size 5x5
	#choose the best features via pooling
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#randomly turn neurons on and off to improve convergence
	# model.add(Dropout(0.25))
	#flatten since too many dimensions, we only want a classification output
	model.add(Flatten())
	#fully connected to get all relevant data
	model.add(Dense(32, activation='relu'))
	#one more dropout for convergence' sake :) 
	# model.add(Dropout(0.5))
	#output a softmax to squash the matrix into output probabilities
	model.add(Dense(num_category, activation='softmax'))
	return model

def fit_and_evaluate(t_x, val_x, t_y, val_y,X_test,y_test, EPOCHS, BATCH_SIZE,opt,input_shape,num_category):
    
    ##model building
	model = create_model(input_shape,num_category)

	model.compile(loss=keras.losses.categorical_crossentropy,optimizer=opt,metrics=['accuracy'])

	results = model.fit(t_x, t_y, epochs=EPOCHS, batch_size=BATCH_SIZE,verbose=1, validation_data=(val_x, val_y))  
	output = []
	test_accuracy = model.evaluate(X_test, y_test)
	print("Test Accuracy: ", test_accuracy[1])
	output.append(model.predict_classes(X_test))
	Output_CSV(output[0])
	return results.history


def main():

	X_train, y_train, X_test, y_test, input_shape = load_mnist()

	#set number of categories
	num_category = 10
	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_category)
	y_test = keras.utils.to_categorical(y_test, num_category)

	# One fold Cross Validation
	n_folds=1
	epochs=5
	batch_size=128
	lr = 0.001

	model_history = [] 

	for i in range(n_folds):
		print("Training on Fold: ",i+1)
		opt = Adam(lr=0.001)
		t_x, val_x, t_y, val_y = train_test_split(X_train, y_train, test_size=0.25)
		model_history.append(fit_and_evaluate(t_x, val_x, t_y, val_y,X_test,y_test, epochs, batch_size,opt,input_shape,num_category))
	
	plot(model_history,lr)

	

if __name__ == "__main__":
	main()