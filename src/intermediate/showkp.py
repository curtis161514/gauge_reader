import cv2	
	
def showkpts(filename,kpts):	
	# Reading an image in default mode
	image = cv2.imread('./data/train_images/'+ filename)
	
	# Window name in which image is displayed
	window_name = 'Image'
	
	color = (255, 0, 0)
	
	# Draw line
	#image = cv2.line(image, start_point, end_point, color, thickness)
	image = cv2.circle(image,(int(kpts[0]),int(kpts[1])), 10, (255, 0, 0),5)
	image = cv2.circle(image,(int(kpts[3]),int(kpts[4])), 10, (255, 0, 0),5)
	image = cv2.circle(image,(int(kpts[6]),int(kpts[7])), 10, (255, 0, 0),5)
	image = cv2.circle(image,(int(kpts[9]),int(kpts[10])), 10, (255, 0, 0),5)
	image = cv2.circle(image,(int(kpts[12]),int(kpts[13])), 10, (0, 255, 0),5)
	image = cv2.circle(image,(int(kpts[15]),int(kpts[16])), 10, (0, 255, 0),5)
	image = cv2.circle(image,(int(kpts[18]),int(kpts[19])), 10,(0, 0, 255),5)
	image = cv2.circle(image,(int(kpts[21]),int(kpts[12])), 10, (0, 255, 0),5)

	# Displaying the image 
	cv2.imwrite('raw_'+filename, image)
	cv2.imshow(window_name, image) 
	cv2.waitKey(10000)
	cv2.destroyAllWindows()
	cv2.imwrite(filename, image)


if __name__ == "__main__":
	import json
	with open('./data/train_labels.json', 'r') as label_file:
		label_dict = json.load(label_file)
	#dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
	im = 2
	filename = label_dict['images'][im]['file_name']
	id = label_dict['images'][im]['id']
	kpts = label_dict['annotations'][im]['keypoints']
	showkpts(filename,kpts)
