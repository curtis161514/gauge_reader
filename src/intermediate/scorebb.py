
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from intermediate.showbb import showbb
#compress and load images to np array

global model
model = load_model('models/bb_small.h5')

def scorebb(path_to_image):
	pixels = model.input_shape[1]
	dim = (pixels, pixels)
	img = cv2.imread(path_to_image)
	image_shape = img.shape
	# resize image
	image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	scaled_bb = model.predict(image.reshape(1,pixels,pixels,3))
	
	#unscale bounding box

	unscaled_bb = [scaled_bb[0][0]*image_shape[1],
					scaled_bb[0][1]*image_shape[0],
					scaled_bb[0][2]*image_shape[1],
					scaled_bb[0][3]*image_shape[0]]

	showbb(path_to_image,unscaled_bb)

if __name__ =="__main__":
	path_to_image = 'data/my_images/IMG_0450.jpg'
	scorebb(path_to_image)