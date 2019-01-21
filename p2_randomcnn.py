from scipy.misc import imread,imresize
import pickle
import os
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
from numpy import array

model = load_model(os.path.join('CNNMODELS','ResNet50.h5'))
print(model.summary())


def central_resize(img):
	height,width,channels = img.shape
	if width > height:
		l = (width - height) / 2
		r = width - l
		t = 0
		b = height
	else:
		t = (height - width) / 2
		b = height - t
		l = 0
		r = width
	pil_img = Image.fromarray(np.uint8(img))
	pil_img = pil_img.crop((l,t,r,b))
	pil_img = pil_img.resize([224,224],Image.ANTIALIAS)
	pil_img = array(pil_img)
	return pil_img

def get_all_images_features(filename,model):
	imglist = os.listdir(filename)
	input = np.zeros([1,224,224,3])
	features = {}
	for i,title in enumerate(imglist):
		path = os.path.join(filename,title)
		img = imread(path)
		plt.imshow(img)
		plt.show()
		print(img.shape)
		#1. compare the central resized and normal resized images
		#img = imresize(img,[224,224,3])
		img = central_resize(img)
		plt.imshow(img)
		plt.show()
		# plt.imshow(img)
		# plt.show()
		input[0] = img
		feature = model.predict(input)
		features[title.split('.')[0]] = feature[0]
		if (i+1) % 100 == 0:
			print('extract '+str(i+1)+' features.')
	return features
def random_features(filename):
	imglist = os.listdir(filename)
	features = {}
	for i, title in enumerate(imglist):
		feature = np.random.rand(2048)
		features[title.split('.')[0]] = feature
		if (i + 1) % 100 == 0:
			print('extract ' + str(i + 1) + ' features.')
	return features
def main():

	#extract images features
	#features = get_all_images_features('Flicker8k_Dataset',model)
	features = random_features('Flicker8k_Dataset')
	#save images features
	with open('random_fet.pkl','wb') as f:
		pickle.dump(features,f)
if __name__ == '__main__':
	main()