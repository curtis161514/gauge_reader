import numpy as np
import json
import cv2
import math

#import and scale labels from json
def importlabels(json_path):
	with open(json_path, 'r') as label_file:
		label_dict = json.load(label_file)
		#dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])

	#create ordered lists to unscramble convoluted dictionarys
	filenames = []
	bboxs = []
	dial_points = []
	point_proj = []

	#go through list of images and get the image file name and the annotations for them
	for label in label_dict['images']:
		#append file names
		filenames.append(label['file_name'])
		
		id = label['id']
		#append bounding boxes
		bboxs.append(label_dict['annotations'][id]['bbox'])
		
		#append keypoints
		kpts = label_dict['annotations'][id]['keypoints']
		point_proj.append([kpts[0],kpts[1],kpts[3],kpts[4],kpts[6],kpts[7],kpts[9],kpts[10]])
		dial_points.append([kpts[12],kpts[13],kpts[15],kpts[16],kpts[18],kpts[19],kpts[21],kpts[22]])


	return filenames,bboxs,point_proj,dial_points,

def perspecitve_matrix(pp):
	# specify input coordinates for corners of quadrilateral in order TL, TR, BR, BL as x,
	input = np.float32([
					[pp[0],pp[1]], #1
					[pp[6],pp[7]], #4
					[pp[4],pp[5]], #3
					[pp[2],pp[3]]	#2
					])

	# get top and left dimensions and set to output dimensions point prospective
	width = round(math.hypot(input[0,0]-input[1,0], input[0,1]-input[1,1]))
	height = round(math.hypot(input[0,0]-input[3,0], input[0,1]-input[3,1]))
	#print("width:",width, "height:",height)

	# set upper left coordinates for output rectangle
	x = input[0,0]
	y = input[0,1]

	# specify output coordinates for corners of point perspective in order TL, TR, BR, BL as x,
	output = np.float32([[x,y], [x+width-1,y], [x+width-1,y+height-1], [x,y+height-1]])

	# compute perspective matrix
	matrix = cv2.getPerspectiveTransform(input,output)

	return matrix

#load images, and crop them to the bounding box, and warp them.
def load_images_dpts(filenames,pixels,bboxs,point_proj,dial_points):
	#compress and load images to np array
	dim = (pixels, pixels)
	clipped_warped_images = np.zeros((len(bboxs),pixels,pixels,3))
	scaled_warped_dpts = []
	for i in range(len(bboxs)):
		# load image using cv2.imread() method
		img = cv2.imread('./data/train_images/'+filenames[i])
		x1 = int(bboxs[i][0])
		y1 = int(bboxs[i][1])
		x2 = int(bboxs[i][0]+bboxs[i][2])
		y2 = int(bboxs[i][1]+bboxs[i][3])
		img = img[x1:x2,y1:y2,]

		#warp the image to the point projection then append it to images
		hh, ww = img.shape[:2]
		matrix = perspecitve_matrix(point_proj[i])
		imgWarp = cv2.warpPerspective(img, matrix, (ww,hh), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
		clipped_warped_images[i] = cv2.resize(imgWarp, dim, interpolation = cv2.INTER_AREA)
		
		#clip and warp dial points
		min = warp_dpts(matrix,[dial_points[i][0]-y1,dial_points[i][1]-x1])
		max = warp_dpts(matrix,[dial_points[i][2]-y1,dial_points[i][3]-x1])
		center = warp_dpts(matrix,[dial_points[i][4]-y1,dial_points[i][5]-x1])
		dial = warp_dpts(matrix,[dial_points[i][6]-y1,dial_points[i][7]-x1])

		# imgWarp = cv2.circle(imgWarp,(int(min[0]),int(min[1])), 2, (0, 255, 0),2) #min
		# imgWarp = cv2.circle(imgWarp,(int(max[0]),int(max[1])), 2, (0, 255, 0),2) #max
		# imgWarp = cv2.circle(imgWarp,(int(center[0]),int(center[1])), 2,(0, 0, 255),2) #center
		# imgWarp = cv2.circle(imgWarp,(int(dial[0]),int(dial[1])), 2, (0, 255, 255),2) #dial
		# cv2.imwrite(filenames[i], imgWarp)

		#normalize dialpoint to min max of the cropped image
		xscale = imgWarp.shape[0]
		yscale = imgWarp.shape[1]
		s_min = [(min[0]/xscale),(min[1]/yscale)]
		s_max = [(max[0]/xscale),(max[1]/yscale)]
		s_center = [(center[0]/xscale),(center[1]/yscale)]
		s_dial = [(dial[0]/xscale),(dial[1]/yscale)]

		scaled_warped_dpts.append([s_min[0],s_min[1],s_max[0],s_max[1],s_center[0],s_center[1],s_dial[0],s_dial[1]])

	return clipped_warped_images,np.asarray(scaled_warped_dpts)

def warp_dpts(matrix,p):
	px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
	py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
	return [px,py]


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

def scale_xywh(xywh,point_proj):
	#convert to x,y,w,h and scale bounding box targets
	xywh = np.asarray(xywh) / 1024
	norm_point_proj = np.asarray(point_proj) / 1024
	return xywh,norm_point_proj

def load_images_bb_pp(filenames,pixels,xywh,point_proj,roll = False):
	#compress and load images to np array
	dim = (pixels, pixels)
	images = np.zeros((len(xywh),pixels,pixels,3))
	for i in range(len(xywh)):
		# load image using cv2.imread() method
		img = cv2.imread('./data/train_images/'+filenames[i])

		#all the gauges are in the dead center of the image.  we need to move them
		#for better model generalization

		if roll:
			img,xywh[i],point_proj[i] = roll_image(img,xywh[i],point_proj[i])

		# resize image
		images[i] = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

	return images,xywh,point_proj

def roll_image(img,xywh,point_proj):
	max_x = ((xywh[2] - 1040) / 2) - 100
	max_y = ((xywh[3] - 1040) / 2) - 100
	shiftx = int(max_x *(np.random.rand()*2-1))
	shifty = int(max_y *(np.random.rand()*2-1))
	img = np.roll(img, shiftx, axis=1)
	img = np.roll(img, shifty, axis=0)

	#shift image center
	xywh[0] += shiftx
	xywh[1] += shifty

	#shift point projection
	point_proj[0] += shiftx
	point_proj[1] += shifty
	point_proj[2] += shiftx
	point_proj[3] += shifty
	point_proj[4] += shiftx
	point_proj[5] += shifty
	point_proj[6] += shiftx
	point_proj[7] += shifty

	return img,xywh,point_proj

