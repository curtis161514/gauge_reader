import cv2
import numpy as np

try:
	import tflite_runtime.interpreter as tflite
except:
	print('tflite_runtime not installed try pip install tflite_runtime, using tensorflow instead')
	import tensorflow.lite as tflite

try:
	from utils import perspecitve_matrix
except:
	from src.utils import perspecitve_matrix

# Load TFLite model and allocate tensors.
global xy_wh_pp_model
xy_wh_pp_model = tflite.Interpreter(model_path="models/xy_wh_pp.tflite")
xy_wh_pp_model.allocate_tensors()

# Get input and output tensors.
global xy_wh_pp_input_details
xy_wh_pp_input_details = xy_wh_pp_model.get_input_details()
global xy_wh_pp_output_details
xy_wh_pp_output_details = xy_wh_pp_model.get_output_details ()

# Load TFLite model and allocate tensors.
global mm_cd_model
mm_cd_model = tflite.Interpreter (model_path="models/mm_cd.tflite")
mm_cd_model.allocate_tensors ()

# Get input and output tensors.
global mm_cd_input_details
mm_cd_input_details = mm_cd_model.get_input_details()
global mm_cd_output_details
mm_cd_output_details = mm_cd_model. get_output_details()


def make_square(raw_img):
	image_shape = raw_img.shape
	x = image_shape[0]
	y = image_shape[1]
	if x == y:
		pass
	elif x>y:
		diff = int((x-y)/2)
		raw_img = raw_img[diff:-diff,:,]
	else:
		diff = int((y-x)/2)
		raw_img = raw_img[:,diff:-diff,]
	return raw_img


def clip_gauge(square_img,xy_wh_pp_pred):
	pixels = square_img.shape[0]
	scaled_xy = xy_wh_pp_pred[0]
	scaled_wh = xy_wh_pp_pred[2]
	x = scaled_xy[0][1]*pixels
	y = scaled_xy[0][0]*pixels
	w = scaled_wh[0][1]*pixels
	h = scaled_wh[0][0]*pixels

	#crop image to bounding box
	x1 = int(x - w/2)
	y1 = int(y - h/2)

	# represents the bottom right corner of rectangle
	x2 = int(x + w/2)
	y2 = int(y + h/2)

	clipped_img = square_img[x1:x2,y1:y2,]

	return clipped_img

def transpose_gauge(square_img,clipped_img, xy_wh_pp_pred):
	pixels = square_img.shape[0]
	scaled_pp = xy_wh_pp_pred[1][0]
	unscaled_pp = scaled_pp * pixels	

	#calculate perspective matrix
	matrix = perspecitve_matrix(unscaled_pp)
	hh, ww = clipped_img.shape[:2]
	warped_img = cv2.warpPerspective(clipped_img, matrix, (ww,hh), 
							cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
							borderValue=(0,0,0))

	return warped_img

def total_degrees (mm_cd_pred):

	xcenter = mm_cd_pred[1][0][0]
	ycenter = mm_cd_pred[1][0][1]

	#shift the coordinates to where the center of the dial is 0,0
	xmin = mm_cd_pred[0][0][0] - xcenter
	ymin = ycenter - mm_cd_pred[0][0][1]
	xmax = mm_cd_pred[0][0][2] - xcenter
	ymax = ycenter - mm_cd_pred[0][0][3]


	a = ((xmin-xmax)**2 + (ymin-ymax)**2)**(1/2) #length between scale min and scale max
	b = ((0-xmin)**2 + (0-ymin)**2)**(1/2) #length between center and scale min
	c = ((0-xmax)**2 + (0-ymax)**2)**(1/2) #length between center and scale max

	theta_a = (b**2 + c**2 - a**2) / (2 * b * c) #cosine rule radians

	#if max and min are both in lower quadrant then definately > 180
	if ymin < 0 and ymax < 0:
		theta_a = 360-round(np.degrees(np.arccos(theta_a)),2) #convert to degrees
	
	#if min is lower and max is upper need to check if its more than 180
	elif (ymin < 0 and ymax > 0) and (abs(ymin) > abs(ymax)):
		theta_a = 360-round(np.degrees(np.arccos(theta_a)),2) #convert to degrees
	
	#if min is upper and max is lower need to check if its more than 180
	elif (ymin > 0 and ymax < 0) and (abs(ymin) < abs(ymax)):
		theta_a = 360-round(np.degrees(np.arccos(theta_a)),2) #convert to degrees
		
	else: #its less than 180
		theta_a = round(np.degrees(np.arccos(theta_a)),2) #convert to degrees

	return theta_a

def pointer_degrees (mm_cd_pred):

	xcenter = mm_cd_pred[1][0][0]
	ycenter = mm_cd_pred[1][0][1]

	#shift the coordinates to where the center of the dial is 0,0
	xmin = mm_cd_pred[0][0][0] - xcenter
	ymin = ycenter - mm_cd_pred[0][0][1] 
	xdial = mm_cd_pred[1][0][2] - xcenter
	ydial = ycenter - mm_cd_pred[1][0][3]

	a = ((xmin-xdial)**2 + (ymin-ydial)**2)**(1/2) #length between scale min and scale max
	b = ((0-xmin)**2 + (0-ymin)**2)**(1/2) #length between center and scale min
	c = ((0-xdial)**2 + (0-ydial)**2)**(1/2) #length between center and scale max

	theta_a = (b**2 + c**2 - a**2) / (2 * b * c) #cosine rule radians

	#if ymin and ydial are both in opposite lower quadrants
	if ymin < 0 and ydial < 0 and xmin < 0 and xdial > 0 :
		theta_a = 360-round(np.degrees(np.arccos(theta_a)),2) #convert to degrees
	
	#if min is lower and max is upper need to check if its more than 180
	elif ymin < 0 and ydial > 0 and xdial > 0 and (abs(ymin) > abs(ydial)):
		theta_a = 360-round(np.degrees(np.arccos(theta_a)),2) #convert to degrees
	
	#if min is lower and max is upper need to check if its more than 180
	elif ymin > 0 and ydial < 0 and abs(ymin) < abs(ydial):
		theta_a = 360-round(np.degrees(np.arccos(theta_a)),2) #convert to degrees
	
	else: #its less than 180
		theta_a = round(np.degrees(np.arccos(theta_a)),2) #convert to degrees

	return theta_a

def annotate_img (imgWarp_rs,mm_cd_pred,reading):
	unnorm_mm = mm_cd_pred[0][0]*imgWarp_rs.shape[0] #unnormalize min and max
	unnorm_cd = mm_cd_pred[1][0]*imgWarp_rs.shape[0] #unnormalize center and dial points
	
	#project dial face
	radius1 = ((unnorm_cd[0]-unnorm_mm[0])**2 + (unnorm_cd[1]-unnorm_mm[1])**2 )**(1/2)
	radius2 = ((unnorm_cd[0]-unnorm_mm[2])**2 + (unnorm_cd[1]-unnorm_mm[3])**2 )**(1/2)
	radius = int((radius1+radius2) / 2)
	imgWarp_rs = cv2.circle(imgWarp_rs,(int(unnorm_cd[0]),int(unnorm_cd[1])), radius, (0, 255, 0),1)

	#draw line from center to pointer
	imgWarp_rs = cv2.line(imgWarp_rs,(int(unnorm_cd[0]),int(unnorm_cd[1])), \
		(int(unnorm_cd[2]),int(unnorm_cd[3])) , (0, 225, 225),1)

	# Draw keypoints
	imgWarp_rs = cv2.circle(imgWarp_rs,(int(unnorm_mm[0]),int(unnorm_mm[1])), 2, (255, 0, 0),2)
	imgWarp_rs = cv2.circle(imgWarp_rs,(int(unnorm_mm[2]),int(unnorm_mm[3])), 2, (255, 0, 0),2)
	imgWarp_rs = cv2.circle(imgWarp_rs,(int(unnorm_cd[0]),int(unnorm_cd[1])), 2, (255, 0, 0),2)
	imgWarp_rs= cv2.circle(imgWarp_rs,(int(unnorm_cd[2]),int(unnorm_cd[3])), 2, (255, 0, 0),2)

	imgWarp_rs = cv2.putText(imgWarp_rs, reading, (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,0,255), 2)
	
	# save the image 
	# cv2.imwrite(savename, imgWarp_rs)
	#print('total degrees ' + str(maxdeg) + ' pointer angel ' + str(pointdeg) + ' gauge percent ' + str(round(pointdeg/maxdeg,2)*100))

	return imgWarp_rs

def score(raw_img):
	# raw_img = cv2.imread(path_to_image)
	# cv2.imwrite('raw.png', raw_img)

	#make the image square so there are no issues compressing
	square_img = make_square(raw_img)
	# cv2.imwrite('square.png', square_img)

	#reshape image to be the same as the model
	model_input_shape = xy_wh_pp_input_details [0]['shape'][1]
	dim = (model_input_shape, model_input_shape)
	rs_image = cv2.resize(square_img, dim, interpolation = cv2.INTER_AREA)
	rs_image = np.float32(rs_image).reshape(1, model_input_shape, model_input_shape, 3)
	xy_wh_pp_model.set_tensor(xy_wh_pp_input_details[0]['index'],rs_image)
	xy_wh_pp_model.invoke()
	#make predictions for each head in the model
	xy_wh_pp_pred = []
	for output in xy_wh_pp_output_details:
		xy_wh_pp_pred.append(xy_wh_pp_model.get_tensor(output['index']))

	#clip the image to the predicted bounding box
	clipped_img = clip_gauge(square_img, xy_wh_pp_pred)
	# cv2.imwrite('clipped.png', clipped_img)

	#warp the image to the point projection
	warped_image = transpose_gauge(square_img,clipped_img, xy_wh_pp_pred)
	# cv2.imwrite('warped.png', warped_image)

	#reshape image to be the same as the model
	model_input_shape = mm_cd_input_details[0]['shape'][1]
	dim = (model_input_shape, model_input_shape)
	imgWarp_rs = cv2. resize (warped_image, dim, interpolation = cv2.INTER_AREA)
	# cv2. imwrite( 'warped_rs.png', imgWarp_rs)

	#make kpts prediction
	mm_cd_model.set_tensor(mm_cd_input_details[0]['index'],
		np.float32(imgWarp_rs.reshape(1,model_input_shape,model_input_shape,3)))
	mm_cd_model. invoke()

	#make predictions for each head in the model
	mm_cd_pred = []
	for output in mm_cd_output_details:
		mm_cd_pred.append (mm_cd_model.get_tensor(output['index']))
	
	#calculate angle from gauge min to gauge max
	maxdeg = total_degrees(mm_cd_pred)

	#calculate angle from gauge min to pointer tip
	pointdeg = pointer_degrees(mm_cd_pred)

	#calculate percentage
	reading = str(round((pointdeg/maxdeg)*100,1)) +'%'

	#annotate the image
	annotated_image = annotate_img(imgWarp_rs,mm_cd_pred,reading)

	return annotated_image

if __name__ == "__main__":

	# define a video capture object
	vid = cv2.VideoCapture('./data/Real Gauges/Gauge Reading/meter_f/meter_f_vid1.mp4')

	while(vid.isOpened()):
		ret, frame = vid.read()
		if not ret:
			break
		scored_frame = score(frame)
		cv2.imshow('scored frame',scored_frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	vid.release()
	cv2.destroyAllWindows()



