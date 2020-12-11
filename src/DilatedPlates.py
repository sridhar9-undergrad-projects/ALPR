import sys
sys.path.append('C:\Python27\Lib\site-packages')
import cv2
import numpy as np
import  os
import pytesseract
from PIL import Image
from ConnectedAnalysis import ConnectedAnalysis
import post_process as pp
input_folder = r"C:\Users\SRIDHAR\Documents\python\final\seg_new";
output_folder= "temp";

def postProcess(str):           #processess plate string
    res=""
    for ind in range(len(str)):
        ch = str[ind]
        if ((ch>='A' and ch<= 'Z') or (ch>='0' and ch<='9')):
            res+=ch
        if (ind>0 and str[ind-1]=='\\' and str[ind]=='n'):
            continue
    return res
i=1
ini_res = []
for filename in os.listdir(input_folder):
  #  print "alkfa"
    img = cv2.imread(os.path.join(input_folder,filename));  #44 #165
    outName=os.path.join(output_folder,filename);
    thresh_image=img;
    finalstr = ""
    finalstr= pytesseract.image_to_string(Image.fromarray(cv2.bitwise_not(img)));
    finalstr = postProcess(finalstr)
    temp = ""
    for ind in range(len(filename)):
        if ind<len(filename)-4:
            temp += filename[ind]
    ini_res.append([int(temp),finalstr])
    print (temp+" "+finalstr);
    i = i+1

#print(i)

pp.result(ini_res)