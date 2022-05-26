import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

def angle_to_value(angles, info):
    g = (info["delta_a"])/(info["maxv"]-info["minv"])
    
    values=[]
    for a in angles:
        values.append(round(((a)*(1/g)) + info["minv"],2))    #linear interpolation
        
    return values

def coords_to_angles(pointer_coords,info_coords):
    c = np.linalg.norm(info_coords[0]-info_coords[2]) #fixed length between center and scale min
    
    angles=[]
    for p in pointer_coords:
        a = np.linalg.norm(p-info_coords[0]) #length between tip and scale min
        b = np.linalg.norm(p-info_coords[2]) #length between tip and center
        
        x = (b**2 + c**2 - a**2) / (2 * b * c) #cosine rule
        
        angles.append(round(np.degrees(np.arccos(x)),2))
        
    a = np.linalg.norm(info_coords[1]-info_coords[0]) #length between scale max and scale min
    b = np.linalg.norm(info_coords[1]-info_coords[2]) #length between scale max and center
    x = (b**2 + c**2 - a**2) / (2 * b * c) #cosine rule
    
    info = {"delta_a": round(360 - np.degrees(np.arccos(x)),2) } #will be incorrect if scale is less that 180 degrees
    
    return info, angles

video_file = r"meter_a\meter_a_vid1.mp4"

cap = cv2.VideoCapture(video_file)
coords=[]
label_coords={0: "",
              1: "",
              2: ""} #0: Scale min, 1: Scale max, 2: Pointer Center
label_count = 0

def onclick(event):
    ix, iy = event.xdata, event.ydata
    coords.append(np.array([ix,iy]))

    implot.figure.canvas.mpl_disconnect(cid)
    
    plt.close()
    
def first_onclick(event):
    ix, iy = event.xdata, event.ydata
    global label_count
    
    if label_count==3:
        coords.append(np.array([ix,iy]))
    else:
        label_coords[label_count] = np.array([ix,iy])
    label_count += 1

    if label_count==4:
        implot.figure.canvas.mpl_disconnect(cid)
        plt.close()

count=0
while(cap.isOpened()):
    print("Frame: ",count)
    
    ret, frame = cap.read()
    
    if ret: #break on last frame
        implot=plt.imshow(frame)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')

        if count==0: #for first frame record extra coords
            print("ENTER: Scale min, Scale max, Pointer Center")
            cid = implot.figure.canvas.mpl_connect('button_press_event', first_onclick)
            plt.show()
        else:
            cid = implot.figure.canvas.mpl_connect('button_press_event', onclick)
            plt.show()
    else:
        break
    
    count+=1
    
cap.release()
cv2.destroyAllWindows()

print("Calculating angles ...")
info, angles = coords_to_angles(coords, label_coords)
print("COMPLETE \n")

info["minv"] = float(input("Minimum Value of Scale: "))
info["maxv"] = float(input("Maximum Value of Scale: "))

print("\nCalculating values ...")
values = angle_to_value(angles, info)
print("COMPLETE \n")

print("Saving to json ...")
annotations = {"angles": angles,
               "values": values,
               "info": info}

f = open("annotations.json","w")
json.dump(annotations, f)
f.close() 

print("COMPLETE. Please rename and move json file")
