from matplotlib import scale
import numpy as np
import json
import cv2

#import and scale labels from json
def importlabels(json_path):
	with open(json_path, 'r') as label_file:
		label_dict = json.load(label_file)
		#dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])


	#create ordered lists to unscramble convoluted dictionarys
	filenames = []
	bboxs = []
	# dial_points = []
	# point_proj = []

	#go through list of images and get the image file name and the annotations for them
	for label in label_dict['images']:
		#append file names
		filenames.append(label['file_name'])
		
		id = label['id']
		#append bounding boxes
		bboxs.append(label_dict['annotations'][id]['bbox'])
		
		#append keypoints
		# kpts = label_dict['annotations'][id]['keypoints']
		# point_proj.append([kpts[0],kpts[1],kpts[3],kpts[4],kpts[6],kpts[7],kpts[9],kpts[10]])
		# dial_points.append([kpts[12],kpts[13],kpts[15],kpts[16],kpts[18],kpts[19],kpts[21],kpts[22]])


	return filenames,bboxs #,dial_points,point_proj

def bbox_to_xywh (bboxs,numimages):
	#convert to x,y,w,h and scale bounding box targets
	xywh = []
	for bbox in bboxs[0:numimages]:
		x1 = bbox[0]
		y1 = bbox[1]
		x2 = bbox[0] + bbox[2]
		y2 = bbox[1] + bbox[3]

		x = (x1 + x2)/2
		y = (y1 + y2)/2
		w = x2 - x1
		h = y2-y1
		xywh.append([x,y,w,h])
	return xywh

def scale_xywh (xywh):
	#convert to x,y,w,h and scale bounding box targets
	xywh = np.asarray(xywh) / 1024
	return xywh

def load_images(filenames,pixels,xywh,roll = False):
	#compress and load images to np array
	dim = (pixels, pixels)
	images = np.zeros((len(xywh),pixels,pixels,3))
	for i in range(len(xywh)):
		# load image using cv2.imread() method
		img = cv2.imread('./data/train_images/'+filenames[i])

		if roll:
			max_x = (xywh[i][2] - 1040) / 2
			max_y = (xywh[i][3] - 1040) / 2
			shiftx = int(max_x *(np.random.rand()*2-1))
			shifty = int(max_y *(np.random.rand()*2-1))
			img = np.roll(img, shiftx, axis=1)
			img = np.roll(img, shifty, axis=0)

			#correct image center
			xywh[i][0] += shiftx
			xywh[i][1] += shifty
		# resize image
		images[i] = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	return images,xywh

###############################################################
#-------------Train Model--------------------------------------
###############################################################
#load labels from json
filenames,bboxs = importlabels('data/train_labels.json')
numimages = 10000 #number of images to use
pixels = 224 # pxp scale
xywh = bbox_to_xywh(bboxs,numimages)
images,xywh = load_images(filenames,pixels,xywh,roll=True)
norm_xywh = scale_xywh(xywh)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, BatchNormalization
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small

import tensorflow as tf
tf.keras.backend.clear_session()

pretrained_model = MobileNetV3Large(input_shape=(pixels,pixels,3), #classes=4,
                             weights="imagenet", pooling=None, include_top=False)
#pretrained_model.summary()
# set all layers trainable because when I froze most of the layers the model didn't learn so well
# for layer in pretrained_model.layers:
#     #layer.trainable = True
# 	print(layer)
#pretrained_model.layers[-1].trainable = True
last_output = pretrained_model.layers[-1].output
x = GlobalMaxPooling2D()(last_output)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(4, activation='linear')(x)
model = Model(pretrained_model.input, x)

model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
model.summary()
# val_acc with min_delta 0.003; val_loss with min_delta 0.01
plateau = ReduceLROnPlateau(monitor="loss", mode="min", patience=4,
                            min_lr=1e-6, factor=0.3, min_delta=0.001,
                            verbose=0)
checkpointer = ModelCheckpoint(filepath='./models/bb_large.h5', verbose=0, save_best_only=False,
                               monitor="val_accuracy", mode="max",
                               save_weights_only=False)

for e in range(20):
	print(e)
	train = np.random.choice(9000,500)
	val = np.random.choice(1000,100) + 9000
	model.fit(x=images[train],y=norm_xywh[train],batch_size=10,epochs=5,verbose='auto',
    		callbacks=[checkpointer,plateau],validation_data = (images[val],norm_xywh[val]))

model.predict(images[10].reshape(1,224,224,3))
norm_xywh[10]