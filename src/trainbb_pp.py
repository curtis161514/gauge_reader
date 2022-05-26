from matplotlib import scale
import numpy as np
import json
import cv2
from src.utils import importlabels,bbox_to_xywh,load_images_bb_pp,scale_xywh

filenames,bboxs,point_proj,dial_points = importlabels('data/train_labels.json')
numimages = 10000 #number of images to use
pixels = 224 # pxp scale
xywh = bbox_to_xywh(bboxs,numimages)

#load all images and roll them some
#you need 16gb memory
images,xywh,point_proj = load_images_bb_pp(filenames,pixels,xywh,point_proj,roll=True)

#normalize shifted targets
norm_xywh,norm_point_proj = scale_xywh(xywh,point_proj)


from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, BatchNormalization
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small

import tensorflow as tf
tf.keras.backend.clear_session()

pretrained_model = MobileNetV3Small(input_shape=(pixels,pixels,3), #classes=4,
                             weights="imagenet", pooling=None, include_top=False)
#pretrained_model.summary()
# set all layers trainable because when I froze most of the layers the model didn't learn so well
# for layer in pretrained_model.layers:
#     #layer.trainable = True
# 	print(layer)

last_output = pretrained_model.layers[-1].output

#xy classifier
xy = GlobalMaxPooling2D()(last_output)
xy = BatchNormalization()(xy)
xy = Dense(256, activation='sigmoid')(xy)
xy = Dense(256, activation='sigmoid')(xy)
xy = Dense(2, activation='linear')(xy)

#wh classifier
wh = GlobalMaxPooling2D()(last_output)
wh = BatchNormalization()(wh)
wh = Dense(256, activation='sigmoid')(wh)
wh = Dense(256, activation='sigmoid')(wh)
wh = Dense(2, activation='linear')(wh)

#keypoints calssifier
pp = GlobalMaxPooling2D()(last_output)
pp = BatchNormalization()(pp)
pp = Dense(1024, activation='sigmoid')(pp)
pp = Dense(1024, activation='sigmoid')(pp)
pp = Dense(8, activation='linear')(pp)
model = Model(pretrained_model.input, (xy,wh,pp))

model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
model.summary()

plateau = ReduceLROnPlateau(monitor="loss", mode="min", patience=4,
                            min_lr=1e-6, factor=0.3, min_delta=0.001,
                            verbose=0)
checkpointer = ModelCheckpoint(filepath='./models/xy_wh_pp.h5', verbose=0, save_best_only=False,
                               monitor="val_accuracy", mode="max",
                               save_weights_only=False)

for e in range(10):
	print(e)
	#train on a random sample of 500 images
	train = np.random.choice(9500,500)
	val = np.random.choice(500,50) + 9500
	model.fit(x=images[train],
				y=(norm_xywh[train,0:2],norm_xywh[train,2:4],norm_point_proj[train]),
				batch_size=10,epochs=5,verbose='auto',
    			callbacks=[checkpointer,plateau],
				validation_data = (images[val],(norm_xywh[val,0:2],norm_xywh[val,2:4],norm_point_proj[val]))
			)

