import cv2
import numpy as np
import math
import os
import query


input_folder= r"C:\Users\SRIDHAR\Documents\python\final\data_set";
out_folder= r"C:\Users\SRIDHAR\Documents\python\final\plates2";
for filename in os.listdir(input_folder):
    img = cv2.imread(os.path.join(input_folder, filename))
    out_name = os.path.join(out_folder,filename);
    if img is  None:
        continue;
   # img = cv2.imread(in_name)
    #img = cv2.imread("seg1.png")
    #cv2.namedWindow("Original Image",cv2.WINDOW_NORMAL)
    # Creating a Named window to display image
    #cv2.imshow("Original Image",img)
    # Display image
    
    # RGB to Gray scale conversion
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #cv2.namedWindow("Gray Converted Image",cv2.WINDOW_NORMAL)
    # Creating a Named window to display image
    #cv2.imshow("Gray Converted Image",img_gray)
    # Display Image
    
    # Noise removal with iterative bilateral filter(removes noise while preserving edges)
    noise_removal = cv2.bilateralFilter(img_gray,9,75,75)
    #noise_removal = cv2.GaussianBlur(img_gray, (5, 5), 0);
    #cv2.namedWindow("Noise Removed Image",cv2.WINDOW_NORMAL)
    # Creating a Named window to display image
    #cv2.imshow("Noise Removed Image ",noise_removal)
    # Display Image
    
    # Histogram equalisation for better results
    equal_histogram = cv2.equalizeHist(noise_removal)
    #cv2.namedWindow("After Histogram equalisation",cv2.WINDOW_NORMAL)
    # Creating a Named window to display image
    #cv2.imshow("After Histogram equalisation",equal_histogram)
    # Display Image
    #--------------------------
    # Morphological opening with a rectangular structure element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=10)
    #cv2.namedWindow("Morphological opening",cv2.WINDOW_NORMAL)
    # Creating a Named window to display image
    #cv2.imshow("Morphological opening",morph_image)
    # Display Image
    
    # Image subtraction(Subtracting the Morphed image from the histogram equalised Image)
    sub_morp_image = cv2.subtract(equal_histogram,morph_image)
    #cv2.namedWindow("Subtraction image", cv2.WINDOW_NORMAL)
    # Creating a Named window to display image
    #cv2.imshow("Subtraction image ", sub_morp_image)
    # Display Image
    #----------------------------------
    # Thresholding the image
    #thresh_image = cv2.adaptiveThreshold(sub_morp_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0);
    ret,thresh_image = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #cv2.namedWindow("Image after Thresholding",cv2.WINDOW_NORMAL)
    # Creating a Named window to display image
    #cv2.imshow("Image after Thresholding ",thresh_image)
    threshImg_copy = thresh_image.copy()        #used at possible plates through blob
    threshImg_copy = cv2.bitwise_not(threshImg_copy)    #inverting image
    
    # Display Image
    # Applying Canny Edge detection
    canny_image = cv2.Canny(thresh_image,250,255)
    #cv2.namedWindow("Image after applying Canny",cv2.WINDOW_NORMAL)
    # Creating a Named window to display image
    #cv2.imshow("Image after applying Canny ",canny_image)
    # Display Image
    canny_image = cv2.convertScaleAbs(canny_image)
    
    # dilation to strengthen the edges
    kernel = np.ones((3,3), np.uint8)
    # Creating the kernel for dilation
    dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
    #cv2.namedWindow("Dilation", cv2.WINDOW_NORMAL)
    # Creating a Named window to display image
    #cv2.imshow("Dilation", dilated_image)
    dilatedImg_copy = dilated_image.copy()   #used for possible plates
    # Displaying Image
    
    # Finding Contours in the image based on edges
    allContoursImg = dilated_image.copy()
    limitedContoursImg = dilated_image.copy()
    _,contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #-----------for angle---------
    def get_angle(p0, p1=np.array([0,0]), p2=None):
        ''' compute angle (in degrees) for p0p1p2 corner
        Inputs:
            p0,p1,p2 - points in the form of [x,y]
        '''
        if p2 is None:
            p2 = p1 + np.array([1, 0])
        v0 = np.array(p0) - np.array(p1)
        v1 = np.array(p2) - np.array(p1)
    
        angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
        return np.degrees(angle)
    #---------------------------------
    i=0
    possible_plates=[]
    orientation=[]
    for cnt in contours:
        flag=0
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)  #box returns 4 points (box[0] is bottom-most point and then clockwise {If level is same then box[0] is bottom-left point then clockwise})
        cv2.drawContours(allContoursImg,[box],0,(255,255,0),2)
        l14 = math.sqrt((box[3][0]-box[0][0])**2 + (box[3][1]-box[0][1])**2)
        l12 = math.sqrt((box[1][0]-box[0][0])**2 + (box[1][1]-box[0][1])**2)
        # width , height depends on length of two sides (longer side becomes width)
        if l14 >l12:
            width = l14
            height = l12
            #to find angle between horizontal
            p0=[box[3][0],box[3][1]]
            p1=[box[0][0],box[0][1]]
            p2=[box[0][0]+100,box[0][1]]
            flag=0
        if l14<=l12:
            width = l12
            height = l14
            p0=[box[1][0] + 100,box[1][1]]
            p1=[box[1][0],box[1][1]]
            p2=[box[0][0],box[0][1]]
            flag=1
        if height==0:
            continue
        as_ratio = float(width)/height # 1 parameter
        #find angle between horizontal
        
        tmp = get_angle(p0,p1,p2)
        #print (tmp)
        angle = tmp #abs(tmp)  #2 parameter
        area = float(abs(box[0][0]*box[1][1] + box[1][0]*box[2][1] + box[2][0]*box[3][1] + box[3][0]*box[0][1] -
        box[1][0]*box[0][1] - box[2][0]*box[1][1] - box[3][0]*box[2][1] - box[0][0]*box[3][1]))/2
        i=i+1
        
    #uncomment to check range of rectangles
    #    if i>=62 and i<=63:
    #        print (box[0][0], box[0][1])
    #        print (box[1][0], box[1][1])
    #        print (box[2][0], box[2][1])
    #        print (box[3][0], box[3][1])
    #        print (i,width,height,as_ratio,angle,area)
    #        cv2.drawContours(tempImage,[box],0,(255,255,0),2)
    #        cv2.imshow("All contours"+str(i),tempImage)
        
        if ((as_ratio<1 or as_ratio>10) or angle>20 or (area<4000 or area>25000)):
            continue
        image_height, image_width = img_gray.shape
        lowerHalfMetric = box[0][1] / float(image_height)
        if lowerHalfMetric<0.5:
            continue;
        cv2.drawContours(limitedContoursImg,[box],0,(255,0,255),2)
        possible_plates.append(box)
        orientation.append(flag)
        
    #cv2.imshow("limited contours",limitedContoursImg) 
    #cv2.imshow("All contours",allContoursImg)
    #sort plates before iterating using test.py
    print("Num of possible plates------> "+str(len(possible_plates)) + " "+str(filename)+"\n")
    result_plate = img      # used for selecting best plate
    mx_num = 0             # used for selecting best plate
    result_plateIndex =-1;
    for i in range(len(possible_plates)):
        #crop portion of plate into new 200x200 image
        box = query.getPointsInOrder(possible_plates[i],orientation[i])
        #print(box)
        
        pts1 = np.float32([box[0],box[1],box[2],box[3]])  #top-left, top-right, bottom-left, bottom-right
        l1= math.sqrt((box[0][0]-box[2][0])**2 + (box[0][1]-box[2][1])**2)
        l2= math.sqrt((box[0][0]-box[1][0])**2 + (box[0][1]-box[1][1])**2)
        if l1>l2:
            width=l1
            length=l2
        else:
            width=l2
            length=l1
        length = int(length)+10
        width = int(width)+10
        pts2 = np.float32([[0,0],[int(width),0],[0,int(length)],[width,length]])
        outImage = np.zeros((width,length,3),np.uint8)
        M = cv2.getPerspectiveTransform(pts1,pts2)
        #plateImg = cv2.warpPerspective(dilatedImg_copy,M,(int(width),int(length)))      # detect possible char through bounding rect method
        plateImg = cv2.warpPerspective(threshImg_copy,M,(int(width),int(length)))        # detect possible char through blob detection
        #cv2.imshow("plate "+str(i),plateImg)    
        # dilate image to strengthen char
    #    kernel = np.ones((3,3), np.uint8)
    #    plateImg = cv2.dilate(plateImg,kernel,iterations=1)  
         # detect white blobs on black background -----------------------------------------
        
    #    params = cv2.SimpleBlobDetector_Params()
    #    params.filterByArea = True
    #    #params.minArea = int((length/2)*(width/(length)))
    #    params.minArea = 10
    #    params.maxArea = int((length/2)*(width/4))
    #    detector = cv2.SimpleBlobDetector_create(params)
    #    keypoints = detector.detect(plateImg)
    #    im_with_keypoints = cv2.drawKeypoints(plateImg, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #    cv2.imshow("pts "+str(i), im_with_keypoints)
    #    
        
        #-------------------------------------------------------------------------------
        
        # detect possible char with bounding box-------------------------
    #    kernel = np.ones((3,3), np.uint8)
    #    d_image = cv2.dilate(plateImg,kernel,iterations=1)
        image_copy = plateImg.copy()
        new,contours, hierarchy = cv2.findContours(plateImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count=0
        for cnt in contours:
            [x,y,w,h] = cv2.boundingRect(cnt)
            area = w*h
            if (area<50 or area>((length/2)*(width/4))):
                continue
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), (255, 0, 255), 1)
            count = count+1
        #cv2.imshow("plate "+str(i),image_copy) 
        #print("Num of rects in plate ",i," ---> ",count)  
        
        if (count>mx_num):
            mx_num = count
            result_plate = image_copy
            result_plateIndex=i;
        #-----------------------------------------------------------------
    
    #cv2.imshow("possible plate",result_plate)
    i=result_plateIndex;
    if i!=-1:
        box = query.getPointsInOrder(possible_plates[i], orientation[i])
        # print(box)

        pts1 = np.float32([box[0], box[1], box[2], box[3]])  # top-left, top-right, bottom-left, bottom-right
        l1 = math.sqrt((box[0][0] - box[2][0]) ** 2 + (box[0][1] - box[2][1]) ** 2)
        l2 = math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
        if l1 > l2:
            width = l1
            length = l2
        else:
            width = l2
            length = l1
        length = int(length) + 10
        width = int(width) + 10
        pts2 = np.float32([[0, 0], [int(width), 0], [0, int(length)], [width, length]])
        outImage = np.zeros((width, length, 3), np.uint8)
        M = cv2.getPerspectiveTransform(pts1, pts2)
        # plateImg = cv2.warpPerspective(dilatedImg_copy,M,(int(width),int(length)))      # detect possible char through bounding rect method
        result_plate = cv2.warpPerspective(img_gray, M, (int(width), int(length)))
        #thresh_image = cv2.adaptiveThreshold(result_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0);
        cv2.imwrite(out_name,result_plate)
    cv2.waitKey()