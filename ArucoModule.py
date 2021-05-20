import cv2.cv2
import cv2.aruco as aruco
import numpy as np
import os

def findArucoMarkers(img,markerSize=6,totalMarkers=250,draw=True)->list:
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   #convert the image to a grayscale
    # arucoDict=aruco.Dictionary_get(aruco.DICT_6X6_250)
    key=getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict=aruco.Dictionary_get(key)
    arucoParam=aruco.DetectorParameters_create()

    #rejected bounding box is returned when the id is not decoded but the entity is detected as a marker 
    bboxs,ids,rejected_bboxs=aruco.detectMarkers(imgGray,arucoDict,parameters=arucoParam)  

    # print(ids)
    if(draw):
        aruco.drawDetectedMarkers(img,bboxs)
        # print(bboxs)

    return [bboxs,ids]   #bbox and ids are of type list



def augmentAruco(bbox,id,img,augment_image,draw_id_on_image=True):

    top_left=     [bbox[0][0][0],bbox[0][0][1]]
    top_right=    [bbox[0][1][0],bbox[0][1][1]]
    bottom_right= [bbox[0][2][0],bbox[0][2][1]]
    bottom_left=  [bbox[0][3][0],bbox[0][3][1]]
    
    h,w,c=augment_image.shape

    pts1=np.array([top_left,top_right,bottom_right,bottom_left])
    pts2=np.array([[0,0],[w,0],[w,h],[0,h]])

    matrix, _ = cv2.findHomography(pts1,pts2)
    imgOutput = cv2.warpPerspective(augment_image, matrix, (img.shape[1],img.shape[0]))

    return imgOutput




def main():
    cap=cv2.VideoCapture(0)

    while(1):
        success,img=cap.read()

        augImage=cv2.imread("images/img_breaking_wall_small.jpg")
        arucoFound = findArucoMarkers(img)   #get the postition of the bounding boxes and their ids
        
        #Loop through each marker and augment them
        if(len(arucoFound[0])!=0):   #if the length of the bboxs list is 0, then it means that no marker is detected
            # for i in range(len(arucoFound[0])):
            #     print(arucoFound[0][i],arucoFound[1][i])
            for bbox,id in zip(arucoFound[0],arucoFound[1]):
                img=augmentAruco(bbox,id,img,augImage)

                # print(bbox,id)
                # # [[[413. 203.]
                # #   [410. 233.]
                # #   [380. 231.]
                # #   [383. 201.]]] [124]


        
        cv2.imshow("Image",img)
        

        key=cv2.waitKey(1)
        if(key==113):
            break
        



if __name__=="__main__":
    main()


