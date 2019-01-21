from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
def main():
	#get VGG16 model from keras
	model = ResNet50(weights='imagenet')
	print(model.summary())
	#remove the last layers of VGG16 model
	model = Model(inputs=model.inputs,outputs=model.layers[-2].output)
	output = model.output

	#save the cnn model
	model.save(os.path.join('CNNMODELS','ResNet50.h5'))
if __name__ == '__main__':
	main()