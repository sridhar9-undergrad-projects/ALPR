import sys
sys.path.append('C:\Python27\Lib\site-packages')
import cv2
import numpy as np
import  os
import pytesseract
from PIL import Image
class ConnectedAnalysis:
    def labelCompo(self,thresh_image, vis, labelNo, ui, uj):
        vis[ui][uj] = labelNo;
        rows, cols = thresh_image.shape;
        dx = [-1, 0, 1, 0, -1, -1, 1, 1];
        dy = [0, -1, 0, -1, -1, 1, -1, 1];
        for i in range(0, 8):
            xx = ui + dx[i];
            yy = uj + dy[i];
            if xx >= rows or xx < 0 or yy >= cols or yy < 0 or vis[xx][yy] != 0 or thresh_image[xx][yy] == 0:
                continue;
            self.labelCompo(thresh_image, vis, labelNo, xx, yy);

    def connectedCompo(self,thresh_Image):
        rows, cols = thresh_Image.shape;
        vis = np.zeros((rows, cols));
        labelNo = 1;
        for i in range(0, rows):
            for j in range(0, cols):
                if vis[i][j] != 0 or thresh_Image[i][j] == 0:
                    continue;
                self.labelCompo(thresh_Image, vis, labelNo, i, j);
                labelNo += 1;
        demo = thresh_Image.copy();
        return vis;
    def find(self,parent,label):
        if parent[int(label)]!=label:
            newLabel =int(label);
            parent[newLabel]=self.find(parent,parent[newLabel]);
        return  label;
    def findConnected(self,thresh_Image):
        rows,cols = thresh_Image.shape;
        dx = [-1, 0, 1, 0, -1, -1, 1, 1];
        dy = [0, -1, 0, -1, -1, 1, -1, 1];
        labels=np.zeros((rows,cols));
        for i in range(0,rows):
            for j in range(0,cols):
                labels[i][j]=0;
        parent=np.zeros(100000);
        #for i in range(0,100):
         #   parent.append(i);
        labelNo=1;
        parent[labelNo]=labelNo;
        for i in range(0,rows):
            for j in range(0,cols):
                allZero=1;
                for k in range(0,8):
                    xx = i + dx[k];
                    yy = j + dy[k];
                    if xx >= rows or xx < 0 or yy >= cols or yy < 0:
                        continue;
                    if labels[xx][yy]==0 or thresh_Image[xx][yy]==0:
                        continue;
                    allZero-=1;
                if allZero==1:
                    labels[i][j]=labelNo;
                    labelNo+=1;
                    parent[labelNo]=labelNo;
                else:
                    labels[i][j]=labelNo;
                    for k in range(0, 8):
                        xx = i + dx[k];
                        yy = j + dy[k];
                        if xx >= rows or xx < 0 or yy >= cols or yy < 0:
                            continue;
                        if labels[xx][yy] == 0 or thresh_Image[xx][yy] == 0:
                            continue;
                        neighbourLabel =int(labels[i][j]);
                        parent[neighbourLabel]=parent[(int)(labelNo)];
                    labelNo+=1;
                    parent[labelNo]=labelNo;
        #print ("LABELS "+str(labelNo));
        for i in range(0,rows):
            for j in range(0,cols):
                if thresh_Image[i][j]==0:
                    continue;
                labels[i][j]=self.find(parent,labels[i][j]);
        return  labels;