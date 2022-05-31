import cv2	
	
def showbb(path_to_image,xywh):	
	# Reading an image in default mode
	image = cv2.imread(path_to_image)
	
	# Window name in which image is displayed
	window_name = 'Image'
	
	# represents the top left corner of rectangle
	# x1 = int(bbox[0])
	# y1 = int(bbox[1])
	x1 = int(xywh[0] - xywh[2]/2)
	y1 = int(xywh[1] - xywh[3]/2)
	start_point = (x1, y1)

	
	# represents the bottom right corner of rectangle
	# x2 = int(bbox[0]+bbox[2])
	# y2 = int(bbox[1]+bbox[3])
	x2 = int(xywh[0] + xywh[2]/2)
	y2 = int(xywh[1] + xywh[3]/2)

	end_point = (x2, y2)

	# Blue color in BGR
	color = (255, 0, 0)
	
	# Line thickness of 2 px
	thickness = 5
	
	# Draw a rectangle with blue line borders of thickness of 2 px
	image = cv2.rectangle(image, start_point, end_point, color, thickness)
	# Displaying the image 
	cv2.imwrite('bb.png', clipped_img)
	cv2.imshow(window_name, image) 
	cv2.waitKey(10000)
	cv2.destroyAllWindows()

if __name__ == "__main__":

	#code gets a random image and puts bb on image from training data
	import json
	import numpy as np
	with open('./data/train_labels.json', 'r') as label_file:
		label_dict = json.load(label_file)
		#dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])

	im = np.random.choice(10000) #pick a number between 0 and 9999
	filename = label_dict['images'][im]['file_name']
	id = label_dict['images'][im]['id']
	bbox = label_dict['annotations'][im]['bbox']
	
	#convert to x,y,w,h
	x1 = bbox[0]
	y1 = bbox[1]
	x2 = bbox[0] + bbox[2]
	y2 = bbox[1] + bbox[3]

	x = (x1 + x2)/2
	y = (y1 + y2)/2
	w = x2 - x1
	h = y2-y1
	xywh = [x,y,w,h]
	path_to_image = './data/train_images/' + filename
	showbb(path_to_image,xywh)

