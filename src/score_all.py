import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import perspecitve_matrix
import math
#compress and load images to np array

global bb_pp_model
xy_wh_pp_model = load_model('models/xy_wh_pp.h5')

global kpts_model
kpts_model = load_model('models/mm_cd.h5')

def score(path_to_image,savename):
	pixels = xy_wh_pp_model.input_shape[1]
	dim = (pixels, pixels)
	raw_img = cv2.imread(path_to_image)
	image_shape = raw_img.shape
	image = cv2.resize(raw_img, dim, interpolation = cv2.INTER_AREA)
	xy_wh_pp_pred = xy_wh_pp_model.predict(image.reshape(1,pixels,pixels,3))
	
	#unscale bounding box
	scaled_xy = xy_wh_pp_pred[0]
	scaled_wh = xy_wh_pp_pred[1]
	unscaled_xywh = [scaled_xy[0][0]*image_shape[1],
					scaled_xy[0][1]*image_shape[0],
					scaled_wh[0][0]*image_shape[1],
					scaled_wh[0][1]*image_shape[0]]

	#crop image to bounding box
	y1 = int(unscaled_xywh[0] - unscaled_xywh[2]/2)
	x1 = int(unscaled_xywh[1] - unscaled_xywh[3]/2)

	# represents the bottom right corner of rectangle
	y2 = int(unscaled_xywh[0] + unscaled_xywh[2]/2)
	x2 = int(unscaled_xywh[1] + unscaled_xywh[3]/2)

	clipped_img = raw_img[x1:x2,y1:y2,]

	# cv2.imwrite('clipped.png', clipped_img)
	# cv2.imshow('pred', clipped_img) 
	# cv2.waitKey(10000)
	# cv2.destroyAllWindows()

	#unscale point_projections
	scaled_pp = xy_wh_pp_pred[2][0]
	unscaled_pp = [scaled_pp[0]*image_shape[1],
				scaled_pp[1]*image_shape[0],
				scaled_pp[2]*image_shape[1],
				scaled_pp[3]*image_shape[0],
				scaled_pp[4]*image_shape[1],
				scaled_pp[5]*image_shape[0],
				scaled_pp[6]*image_shape[1],
				scaled_pp[7]*image_shape[0]]	

	#calculate perspective matrix
	matrix = perspecitve_matrix(unscaled_pp)
	hh, ww = clipped_img.shape[:2]
	imgWarp = cv2.warpPerspective(clipped_img, matrix, (ww,hh), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
	imgWarp_rs = cv2.resize(imgWarp, dim, interpolation = cv2.INTER_AREA)

	#make kpts prediction
	mm_cd_pred = kpts_model.predict(imgWarp_rs.reshape(1,pixels,pixels,3))
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


	maxdeg = total_degrees(mm_cd_pred)
	pointdeg = pointer_degrees(mm_cd_pred)
	reading = str(round((pointdeg/maxdeg)*100,1)) +'%'

	imgWarp_rs = cv2.putText(imgWarp_rs, reading, (150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,0,255), 2)
	
	# save the image 
	cv2.imwrite(savename, imgWarp_rs)
	#print('total degrees ' + str(maxdeg) + ' pointer angel ' + str(pointdeg) + ' gauge percent ' + str(round(pointdeg/maxdeg,2)*100))

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
	
	#if min is lower and max is upper need to check if its more than 180
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

if __name__ =="__main__":
	import os
	import time
	directory = 'data/validation_images/'
	save_directory = 'data/scored_images/'
	if not os.path.exists(save_directory):
		os.mkdir(save_directory)
	
	files = os.listdir(directory)
	
	starttime = time.perf_counter()
	for file in files:
		print(file)
		score(directory+file,save_directory+file)
	endtime = time.perf_counter()

	infrencerate = (endtime - starttime)/len(files)
	print('infrence rate = ' + str(round(infrencerate,2)) + 's per image')
