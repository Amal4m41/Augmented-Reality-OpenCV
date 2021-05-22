import cv2

import ArucoModule as arm



cap=cv2.VideoCapture(0)
# augImage=cv2.imread("images/img_messi.jpg")
augDict=arm.loadImagesToAugment("images")

while(1):
    success,img=cap.read()

    arucoFound = arm.findArucoMarkers(img)   #get the postition of the bounding boxes and their ids
    
    #Loop through each marker and augment them
    if(len(arucoFound[0])!=0):   #if the length of the bboxs list is 0, then it means that no marker is detected
        # for i in range(len(arucoFound[0])):
        #     print(arucoFound[0][i],arucoFound[1][i])
        for bbox,id in zip(arucoFound[0],arucoFound[1]):
            if(id[0] in augDict.keys()):
                img=arm.augmentAruco(bbox,id,img,augDict[id[0]])

            # print(bbox,id)
            # # [[[413. 203.]
            # #   [410. 233.]
            # #   [380. 231.]
            # #   [383. 201.]]] [124]

    cv2.imshow("Image",img)
    
    key=cv2.waitKey(1)
    if(key==113):
        break