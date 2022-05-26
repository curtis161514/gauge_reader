from src.utils import *

#load labels from json
filenames,bboxs,point_proj,dial_points = importlabels('data/train_labels.json')
pixels = 224 # pxp scale
images,targets = load_images_dpts(filenames,pixels,bboxs,point_proj,dial_points)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, BatchNormalization
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small

import tensorflow as tf
tf.keras.backend.clear_session()

pretrained_model = MobileNetV3Small(input_shape=(pixels,pixels,3), #classes=4,
                             weights="imagenet", pooling=None, include_top=False)

# pretrained_model.layers[-1].trainable=True
# pretrained_model.layers[-2].trainable=True
# pretrained_model.layers[-3].trainable=True
# pretrained_model.layers[-4].trainable=True
# pretrained_model.layers[-5].trainable=True
# pretrained_model.layers[-6].trainable=True
last_output = pretrained_model.layers[-1].output

#min_max regression
mm = GlobalMaxPooling2D()(last_output)
mm = BatchNormalization()(dpts)
mm = Dense(512, activation='sigmoid')(mm)
mm = Dense(512, activation='sigmoid')(mm)
mm = Dense(4, activation='linear')(mm)


#center and dial point regression
cd = GlobalMaxPooling2D()(last_output)
cd = BatchNormalization()(cd)
cd = Dense(512, activation='sigmoid')(cd)
cd = Dense(512, activation='sigmoid')(cd)
cd = Dense(4, activation='linear')(cd)

model = Model(pretrained_model.input, (mm,cd))


model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
model.summary()

plateau = ReduceLROnPlateau(monitor="loss", mode="min", patience=4,
                            min_lr=1e-5, factor=0.01, min_delta=0.01,
                            verbose=0)
checkpointer = ModelCheckpoint(filepath='./models/mm_cd.h5', verbose=0, save_best_only=False,
                               monitor="val_accuracy", mode="max",
                               save_weights_only=False)

for e in range(20):
	print(e)
	#train on a random sample of 500 images
	train = np.random.choice(9500,500)
	val = np.random.choice(500,50) + 9500
	model.fit(x=images[train],
				y=(targets[train,0:4],targets[train,4:8]),
				batch_size=10,epochs=5,verbose='auto',
    			callbacks=[checkpointer,plateau],
				validation_data = (images[val],(targets[val,0:4],targets[val,4:8]))
			)

