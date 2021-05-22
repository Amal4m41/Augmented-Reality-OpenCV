import cv2

import ArucoModule as arm



cap=cv2.VideoCapture(0)

augDict=arm.loadImagesToAugment("images")

while(1):
    success,img=cap.read()

    arucoFound = arm.findArucoMarkers(img)   

    if(len(arucoFound[0])!=0):   

        for bbox,id in zip(arucoFound[0],arucoFound[1]):
            if(id[0] in augDict.keys()):
                img=arm.augmentAruco(bbox,id,img,augDict[id[0]])

    cv2.imshow("Image",img)
    
    key=cv2.waitKey(1)
    if(key==113):
        break