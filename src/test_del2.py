import cv2
import os

# prev_list = os.listdir(r"C:\Users\SRIDHAR\Documents\python\final\data_set")
#
# f = open('actual_plates_v2.txt','r')
# lines = [line.rstrip('\n') for line in f]
# plate_map ={}
# for s in lines:
#     flag=0
#     temp = ""
#     i=0
#     while(i<len(s) and s[i]!=' '):
#         temp += s[i]
#         i+=1
#     num = int(temp)
#     #num = temp
#     temp = ""
#     i += 1
#     while (i<len(s)):
#         temp += s[i]
#         i += 1
#     plate_map[num] = temp
#
# i=1
# for s in prev_list:
#     old_name = os.getcwd()+'\\as\\'+s
#     temp =""
#     ind=0
#     l = len(s)
#     while(ind<l and s[ind]!='.'):
#         temp+= s[ind]
#         ind+=1
#     print(i,plate_map[int(temp)])
#     new_name = os.getcwd()+'\\as_in\\'+str(i)+'.jpg'
#     os.rename(old_name,new_name)
#     i+=1






# #
# f1 = open('req_plates_1_repce.txt','r')
# f2 = open('req_plates_no_output.txt','r')
# f3 = open('req_plates_ori_final.txt','r')
# f4 = open('req_plates_wrong_detc.txt','r')
#
# lines1 = [line.rstrip('\n') for line in f1]
# lines2 = [line.rstrip('\n') for line in f2]
# lines3 = [line.rstrip('\n') for line in f3]
# lines4 = [line.rstrip('\n') for line in f4]
#
# for l in lines1:
#     img = cv2.imread('input/'+l+'.jpg')
#     cv2.imwrite('as/'+l+'.jpg',img)
# for l in lines2:
#     img = cv2.imread('input/'+l+'.jpg')
#     cv2.imwrite('final/data_set/'+l+'.jpg',img)
# for l in lines3:
#     img = cv2.imread('input/'+l+'.jpg')
#     cv2.imwrite('final/data_set/'+l+'.jpg',img)
# for l in lines4:
#     img = cv2.imread('input/'+l+'.jpg')
#     cv2.imwrite('final/data_set/'+l+'.jpg',img)
#
#
# f1.close()
# f2.close()
# f3.close()
# f4.close()
