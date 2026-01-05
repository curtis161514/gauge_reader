# Gauge Reader
A computer vision project for reading analog gauges from images or video using machine learning trained on synthetic data.  It is an implementation of this paper. 

https://openaccess.thecvf.com/content/WACV2024/papers/Leon-Alcazar_Learning_to_Read_Analog_Gauges_from_Synthetic_Data_WACV_2024_paper.pdf

All data was provided by the original authors.

# Models
There are two models, xy_wh_pp.tflite and mm_cd.tflite.  The first model outputs the xy cordinates of the center of the gauge,the width and height of it, and a point projection that tells the objects orentation with respect to the camera.

# Running
score_video_tflite.py will constantly read a a video stream and output the gauge reading.  There is no OCR reading the dial numbers.  The Gauge min and max is input manually into the score script.
