
def getPointsInOrder(box,flag):
    '''
    returns in top-left, top-right, bottom-left, bottom-right order 
    '''
    ret = []    
    if flag==0:
        ret = [box[1],box[2],box[0],box[3]]
    else:
        ret = [box[2],box[3],box[1],box[0]]
        
    return ret