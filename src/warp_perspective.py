import numpy as np
import cv2
import math
	
def warp(filename,kpts):	
	
	# Reading an image in default mode
	img = cv2.imread('./data/train_images/'+ filename)

	# Window name in which image is displayed
	window_name = 'Image'
	
	# Draw circles
	#img = cv2.line(image, start_point, end_point, color, thickness)
	img = cv2.circle(img,(int(kpts[0]),int(kpts[1])), 10, (255, 0, 0),5) #tl
	img = cv2.circle(img,(int(kpts[3]),int(kpts[4])), 10, (255, 0, 0),5) #bl
	img = cv2.circle(img,(int(kpts[6]),int(kpts[7])), 10, (255, 0, 0),5) #br
	img = cv2.circle(img,(int(kpts[9]),int(kpts[10])), 10, (255, 0, 0),5) #tr
	img = cv2.circle(img,(int(kpts[12]),int(kpts[13])), 0, (0, 255, 0),5) #min
	img = cv2.circle(img,(int(kpts[15]),int(kpts[16])), 0, (0, 255, 0),5) #max
	img = cv2.circle(img,(int(kpts[18]),int(kpts[19])), 0,(0, 0, 255),5) #center
	img = cv2.circle(img,(int(kpts[21]),int(kpts[22])), 0, (0, 255, 255),5) #dial

	# save the warped output
	cv2.imwrite("actual.jpg", img)
	# Displaying the image 
	cv2.imshow(window_name, img) 
	cv2.waitKey(1000)
	cv2.destroyAllWindows()
	
	hh, ww = img.shape[:2]

	# specify input coordinates for corners of quadrilateral in order TL, TR, BR, BL as x,
	input = np.float32([
					[kpts[0],kpts[1]], 
					[kpts[9],kpts[10]],
					[kpts[6],kpts[7]], 
					[kpts[3],kpts[4]]
					])

	# get top and left dimensions and set to output dimensions point prospective
	width = round(math.hypot(input[0,0]-input[1,0], input[0,1]-input[1,1]))
	height = round(math.hypot(input[0,0]-input[3,0], input[0,1]-input[3,1]))
	print("width:",width, "height:",height)

	# set upper left coordinates for output rectangle
	x = input[0,0]
	y = input[0,1]

	# specify output coordinates for corners of point perspective in order TL, TR, BR, BL as x,
	output = np.float32([[x,y], [x+width-1,y], [x+width-1,y+height-1], [x,y+height-1]])

	# compute perspective matrix
	matrix = cv2.getPerspectiveTransform(input,output)
	print(matrix)

	# do perspective transformation setting area outside input to black
	# Note that output size is the same as the input image size
	imgOutput = cv2.warpPerspective(img, matrix, (ww,hh), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

	# save the warped output
	cv2.imwrite("warped.jpg", imgOutput)

	# show the result
	cv2.imshow("warped image", imgOutput)
	cv2.waitKey(1000)
	cv2.destroyAllWindows()




if __name__ == "__main__":
	import json
	with open('./data/train_labels.json', 'r') as label_file:
		label_dict = json.load(label_file)
	#dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
	im = 5559
	filename = label_dict['images'][im]['file_name']
	id = label_dict['images'][im]['id']
	kpts = label_dict['annotations'][im]['keypoints']
	warp(filename,kpts)
