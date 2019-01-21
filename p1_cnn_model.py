from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.models import load_model
from keras.utils import plot_model
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
def main():
	#get VGG16 model from keras
	model = InceptionV3()
	plot_model(model,'model.png',show_shapes=True)
	#model = load_model(os.path.join('CNNMODELS','baseline.h5'))
	print(model.summary())
	#remove the last layers of VGG16 model
	model = Model(inputs=model.inputs,outputs=model.layers[-2].output)
	#save the cnn model
	model.save(os.path.join('CNNMODELS','InceptionV3.h5'))
if __name__ == '__main__':
	main()