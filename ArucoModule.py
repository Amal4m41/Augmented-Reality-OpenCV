import cv2.cv2
import cv2.aruco as aruco
import numpy as np
import os

def findArucoMarkers(img,markerSize=6,totalMarkers=250,draw=True):
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   #convert the image to a grayscale
    # arucoDict=aruco.Dictionary_get(aruco.DICT_6X6_250)
    key=getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict=aruco.Dictionary_get(key)
    arucoParam=aruco.DetectorParameters_create()
    bbox,ids,rejected_Box=aruco.detectMarkers(imgGray,arucoDict,parameters=arucoParam)

    print(ids)




def main():
    cap=cv2.VideoCapture(0)

    while(1):
        success,img=cap.read()
        cv2.imshow("Image",img)
        cv2.waitKey(1)
        findArucoMarkers(img)



if __name__=="__main__":
    main()


