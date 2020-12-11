import cv2
def lcs(X , Y):
	# find the length of the strings
	m = len(X)
	n = len(Y)

	# declaring the array for storing the dp values
	L = [[None]*(n+1) for i in range(m+1)]

	"""Following steps build L[m+1][n+1] in bottom up fashion
	Note: L[i][j] contains length of LCS of X[0..i-1]
	and Y[0..j-1]"""
	for i in range(m+1):
		for j in range(n+1):
			if i == 0 or j == 0 :
				L[i][j] = 0
			elif X[i-1] == Y[j-1]:
				L[i][j] = L[i-1][j-1]+1
			else:
				L[i][j] = max(L[i-1][j] , L[i][j-1])

	# L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
	return L[m][n]

def result(ini_out):
    f = open('original_plate_numbers.txt','r')            #original (num, lic_plate)
    lines_ori = [line.rstrip('\n') for line in f]
    ori = {}
    out = {}
    for s in lines_ori:
        temp=""
        i=0
        while (i<len(s) and s[i]!=' '):
            temp += s[i]
            i+=1
        i+=1
        num = int(temp)
        temp=""
        while (i<len(s)):
            temp += s[i]
            i+=1
        ori[num] = temp

    for s in ini_out:
        out[s[0]] = s[1]

    for i in range(len(ori)):
        if(out.get(i+1)==None):
            out[i+1] = ''
    print(out)
    print(ori)
    res = []
    count=0
    correct=0
    total=0
    for key,value in ori.items():
        num = lcs(value,out[key])
        correct += num
        total += len(value)
        if (value==out[key]):
            count+=1
            #print(key)
            # img = cv2.imread('plates/'+str(key)+'.jpg')
            # cv2.imwrite('list/exact/'+str(key)+'.jpg',img)
        res.append([num,len(value)])
        # if(len(value)-num==1):
        #     img = cv2.imread('plates/'+str(key)+'.jpg')
        #     cv2.imwrite('list/seg_fault/'+str(key)+'.jpg',img)
        #     print(out[key])
    display_method(res)
    print("Exact match -->"+str(count),round((count/len(ori))*100,2))
    print(correct, total)
    f.close()

def display_method(res):
    arr = [0,0,0,0,0,0,0,0,0,0]
    i=1
    for s in res:
        crt = int(s[0])
        total = int(s[1])
        arr[total-crt]+= 1
        num = total-crt
        i+=1

    for i in range(len(arr)):
        print(i,arr[i],round((arr[i]/len(res))*100,2))
