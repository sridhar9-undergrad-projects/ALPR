import sys
sys.path.append('C:\Python27\Lib\site-packages')
import cv2
import numpy as np
import  os
import pytesseract
from PIL import Image
from ConnectedAnalysis import ConnectedAnalysis
input_folder = r"C:\Users\SRIDHAR\Documents\python\final\plates2";
output_folder= r"C:\Users\SRIDHAR\Documents\python\final\seg_new2";


def preprocess(image):
    imageGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    noise_removal = cv2.GaussianBlur(imageGray, (5, 5), 0);
    clahe = cv2.createCLAHE();
    equal_histogram = clahe.apply(noise_removal);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_image = cv2.morphologyEx(equal_histogram, cv2.MORPH_OPEN, kernel, iterations=20)
    sub_morp_image = cv2.subtract(equal_histogram, morph_image)
    ret, thresh_image = cv2.threshold(sub_morp_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return imageGray;
for filename in os.listdir(input_folder):
    img = cv2.imread(os.path.join(input_folder,filename),0);  #44 #165
    outName=os.path.join(output_folder,filename);
    #cv2.imshow("Image",img);
    if img is None:
        continue;
    #noise_removal = cv2.bilateralFilter(img, 9, 75, 75)


    noise_removal = cv2.GaussianBlur(img, (5, 5), 0);
    clahe = cv2.createCLAHE();
    equal_histogram = clahe.apply(noise_removal);
    #equal_histogram = cv2.equalizeHist(noise_removal);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5 ))
    morph_image = cv2.morphologyEx(equal_histogram, cv2.MORPH_OPEN, kernel, iterations=20)
    sub_morp_image = cv2.subtract(equal_histogram, morph_image)
    ret,otsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21,10);
    thresh_image=th3;
  #  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3));
  #  thresh_image = cv2.dilate(thresh_image, kernel, iterations=1);
  #  thresh_image=cv2.bitwise_not(thresh_image);
  #  thresh_image = cv2.dilate(thresh_image, kernel, iterations=1);
  #  thresh_image = cv2.bitwise_not(thresh_image);
   # thresh_image = cv2.erode(thresh_image, kernel, iterations=1);
    #cv2.imshow("equalize",thresh_image);
    cropped = thresh_image;
    rows,cols= cropped.shape;
    #connectedCompo(thresh_image);
        #print  str(rows)+" "+str(cols);
    horizontalProjection=[];
    totalblacks=0;
    totalwhites=0;
    cnt=0;
    threshold =0;
    def ConnectedCompo(Image):
        output = cv2.connectedComponentsWithStats(Image, 8, cv2.CV_32S)
        connAnalysis = ConnectedAnalysis();
        labels =output[1]#connAnalysis.connectedCompo(Image);
        charCandidates = np.zeros(Image.shape,dtype="uint8");
        finalStr="";
        for label in np.unique(labels):
            if label == -1:
                continue;
            labelMask=np.zeros(Image.shape,dtype="uint8");
            labelMask[labels==label]=255;
            #cv2.imshow(str(label),labelMask);
            _,cnts,_ =cv2.findContours(labelMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);
            if len(cnts)>0:
                c = max(cnts,key=cv2.contourArea);
                (boxX,boxY,boxW,boxH)=cv2.boundingRect(c);
                aspectRatio = boxW/float(boxH);
                solidity= cv2.contourArea(c)/float(boxW*boxH);
                heightRatio = boxH/float(Image.shape[0]);
                keepAspectRatio = aspectRatio < 1.0
                keepSolidity = solidity > 0.14
                keepHeight = heightRatio > 0.2 and heightRatio < 0.95
                area = boxW*boxH;
                cv2.rectangle(img, (boxX, boxY), (boxX + boxW, boxY + boxH), (0, 255, 0), 1);
                cv2.drawContours(img, [c], -1, (0, 255, 0), 2);
                #print  label;
                # check to see if the component passes all the tests
                if keepAspectRatio and keepHeight and keepSolidity:
                    # compute the convex hull of the contour and draw it on the character
                    # candidates mask
                    hull = cv2.convexHull(c)
                   # print "Inside";
                    #tempChar=np.zeros(Image.shape,dtype ="uint8");
                    tempChar = np.zeros(Image.shape, dtype="uint8");
                    #  tempChar = cv2.bitwise_and(tempChar,Image);
                    cv2.drawContours(tempChar, [hull], -1, 255, -1);
                    tempChar = cv2.bitwise_and(tempChar, cv2.bitwise_not(otsu));
                    # cv2.imshow(str(label+1000),characters);
                    # cv2.imwrite(os.path.join(output_folder,str(label)+".jpg"),tempChar);
                    tempChar = cv2.bitwise_not(tempChar);
                    totalCharComponents = len(np.unique(cv2.connectedComponentsWithStats(cv2.bitwise_not(tempChar), 4, cv2.CV_32S)[1]));
                    if totalCharComponents == 2:
                        cv2.drawContours(charCandidates, [hull], -1, 255, -1);
                    #cv2.drawContours(charCandidates, [hull], -1, 255, -1)
                    #tempChar = cv2.bitwise_and(tempChar,Image);
                    #cv2.imwrite(os.path.join(output_folder,str(label)+".jpg"),tempChar);
                   # recognisedChar = pytesseract.image_to_string(Image.open(os.path.join(output_folder, str(label) + ".jpg")),config='-psm 10000');
                   # print recognisedChar;
                   # finalStr = finalStr + recognisedChar;
                #charCandidates = segmentation.clear_border(charCandidates);
        #print  finalStr;
    #    cv2.imshow("RECt",img);
        return  charCandidates;
    charCandidates = ConnectedCompo(cv2.bitwise_not(thresh_image));
    #cv2.imshow("Candidates",charCandidates);
    final = cv2.bitwise_and(charCandidates,cv2.bitwise_not(otsu));
    final = cv2.bitwise_not(final);
    #skeleton = bwmorph(final,'skel',inf);
    #cv2.imshow("Only number Plate",final);
    cv2.imwrite(outName,final);
    copyThresh=thresh_image;
    for i in range(0,rows):
        for j in range(0,cols):
            copyThresh[i][j]=final[i][j];
    #cv2.imwrite(os.path.join(output_folder,"first.jpg"),copyThresh);
    #print pytesseract.image_to_string(Image.fromarray(cv2.imread(os.path.join(output_folder,"first.jpg"))));
    cropped = final;
    for i in range(0,rows):
        horizontalProjection.append(0);
        for j in range(0,cols):
            if cropped[i][j]==0:
                horizontalProjection[i] = horizontalProjection[i]+1;
        #if horizontalProjection[i]>threshold or (i==1):
         #   threshold = horizontalProjection[i];
        threshold =threshold +horizontalProjection[i];
    threshold = threshold/float(rows);
    lowerBound=0;
    upperBound=rows+1;
    #threshold=0;
    horizontalProjectionCropped=thresh_image
    #if totalblacks>totalwhites:
    for i in  range(0,rows):
        if (horizontalProjection[i]>=threshold/3 and horizontalProjection[i]<=threshold)or horizontalProjection[i]==0:
            lowerBound = i;
            break;
    for i in range(0,rows):
        if (horizontalProjection[i]>=threshold/3 and horizontalProjection[i]<=threshold) or horizontalProjection[i]==0:
            totalwhites+=1;
            upperBound=i;
        else:
            totalblacks+=1;
    #print "totalblacks " + str(totalblacks) + " " + str(totalwhites);
    #if totalwhites<totalblacks: singlerow else doublerow
            #if lowerBound<upperBound:
             #   horizontalProjectionCropped = cropped[lowerBound:upperBound, 1:cols + 1];
              #  cv2.imshow("After Ho"+str(i), cv2.WINDOW_NORMAL);
              #  cv2.imshow("After Ho"+str(i), horizontalProjectionCropped);
    #for i in range(1,rows):
       # for j in range(1,cols):
       #     if i>=lowerBound and i<=upperBound:
        #        cropped[i][j]=255;
        #    cropped[i][j]=0;
    horizontalProjectionCropped = cropped[lowerBound:upperBound,1:cols+1];
    #cv2.imshow("After Horizontal",cv2.WINDOW_NORMAL);
    #cv2.imshow("After Horizontal",horizontalProjectionCropped);
    #connectedCompo(horizontalProjectionCropped)
    rows,cols = horizontalProjectionCropped.shape;
    verticalProjection =[];
    verticalProjection.append(0);
    #print "Threshold "+str(threshold)+" "+str(horizontalProjection[lowerBound])+" "+str(upperBound)+" "+str(rows)+"\n";
    threshold =-1;
    maxi =-1;
    for i in range(1,cols):
        verticalProjection.append(0);
        for j in range(1,rows):
            if horizontalProjectionCropped[j][i]==0:
                verticalProjection[i]=verticalProjection[i]+1;
        if verticalProjection[i]<threshold or i==1:
            threshold = verticalProjection[i];
        if verticalProjection[i]>maxi:
            maxi = verticalProjection[i];
        #if verticalProjection[i]==0:
           # print "Zero "+str(i)+"\n";
    startCropping =1;
    characterEncounter=0;
    image_height,image_width = img.shape;
    # "vertical "+str(threshold)+"\n";
    finalStr="";
    for i in range(1,cols):
        if threshold >= verticalProjection[i] and characterEncounter == 1:
            character = final[0:image_height,startCropping:i+2];
            #cv2.imwrite(os.path.join(output_folder, str(i) + ".jpg"), character);
            character =  cv2.imread(os.path.join(output_folder,str(i)+".jpg"));
            break;
            prepocessedCharacter = preprocess(character);
            startCropping = i+1;
            characterEncounter=0;
            #cv2.imshow(str(i),cv2.WINDOW_NORMAL);
            #cv2.imshow(str(i),character);
            tempChar = pytesseract.image_to_string(Image.open(os.path.join(output_folder, str(i) + ".jpg")),
                                                   config='-psm 10000');
          #  print  tempChar;
            finalStr = finalStr + tempChar
        if threshold < verticalProjection[i]:
            characterEncounter=1;
    #ccp = cv2.connectedComponents(thresh_image,labels,4, cv2.CV_16U);
    #cv2.imshow("labels",labels);
    #print  "License PlATE ----> "+finalStr;
    new,contours,heirchy = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE);
    allContours = cv2.drawContours(img,contours,-1,(0,255,0),2)
    #cv2.imshow("Allcontours",img);
cv2.waitKey(0);


