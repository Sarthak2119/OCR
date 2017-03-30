import preprocess as pre
import utility as util
import numpy as np
import cv2

#older one for older dataset
# def pre_init(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = pre.otsu_thresh(img)
#
#     r, c = img.shape
#     cnt0, cnt1 = 0,0
#     for i in range(0,c):
#         if img[0,i] == 0:
#             cnt1 = cnt1+1
#         else:
#             cnt0 = cnt0+1
#         if img[r-1,i] == 0:
#             cnt1 = cnt1+1
#         else:
#             cnt0 = cnt0+1
#
#     for i in range(0,r):
#         if img[i, 0] == 0:
#             cnt1 = cnt1+1
#         else:
#             cnt0 = cnt0+1
#         if img[i, c-1] == 0:
#             cnt1 = cnt1+1
#         else:
#             cnt0 = cnt0+1
#
#     sum = cnt0+cnt1
#     cnt1 = cnt1/sum
#     #print(cnt1)
#     if(cnt1 > 0.5):
#         img = abs(255-img)
#
#     return img

def pre_init(img):
    img = pre.cut_image(img)

    img = pre.thin_image(img)
    return img

#feature 1
def crossing(img):
    cross = []
    # img = abs(1-img)
    # util.display_image(img)
    r,c = img.shape
    x = [int(r/2)]
    y = [int(c/4), int(3*c/4)]

    for i in x:
        cnt = 0
        sum=0
        for j in range(0,c):
            if img[i,j] == 0:
                sum = sum + 1
            else:
                if(sum >= 1):
                    cnt= cnt+1
                sum = 0
        cross.append(cnt)
    for j in y:
        cnt = 0
        sum=0
        for i in range(0,r):
            if img[i,j] == 0:
                sum = sum + 1
            else:
                if(sum >= 1):
                    cnt= cnt+1
                sum = 0
        cross.append(cnt)
    j=0
    cnt=0
    sum = 0
    for i in range(0,r):
        if img[i, j] == 0:
            sum = sum + 1
        else:
            if (sum >= 1):
                cnt += 1
            sum = 0
        j += 1
    cross.append(cnt)

    cnt=0
    sum=0
    j=c-1
    for i in range(0,r):
        if img[i, j] == 0:
            sum = sum + 1
        else:
            if (sum >= 1):
                cnt = cnt + 1
            sum = 0
        j = j - 1
    cross.append(cnt)

    return cross

#feature 2
def histo(img):
    # img = abs(255 - img)
    img = cv2.resize(img, (10, 10), interpolation=cv2.INTER_CUBIC)
    img[img == 255]=1
    # util.display_image(img)
    vec = []
    vsum = np.sum(img, axis=0)
    hsum = np.sum(img, axis=1)
    for i in vsum:
        vec.append(i)
    for i in hsum:
        vec.append(i)
    return vec

#feature 3
def momentum(img):
	momentum_features=[]
	for p in range(0,4):
		for q in range(0,4):
			x1,y1=img.shape
			val = 0
			for x in range(0,x1):
				for y in range(0,y1):
					if img[x][y]!=0:
						val = val+((x**p)*(y**q))
			momentum_features.append(val)
	return momentum_features

#feature 4
def calculate_pixels(img,x_start,x_end,y_start,y_end,val):
    cnt = 0
    for x in range(x_start,x_end):
        for y in range(y_start,y_end):
            if img[x][y]!=val:
                cnt=cnt+1
    return cnt

def zoing_new(img):
    #create 5*5 zoning
    x,y=img.shape
    zone_features = []
    for i in range(0,5):
        for j in range(0,5):
            x1=int(i*x/5)
            y1 = int(j*y/5)
            x2 = int(x1+x/5)
            y2=int(y1+x/5)
            zone_features.append(calculate_pixels(img,x1,x2,y1,y2,0))

    return zone_features

#feature 5
def cnt_hlines(img):
    r,c = img.shape
    l = []
    ans = 0
    for i in range(0,r):
        cnt=0
        for j in range(0,c):
            if img[i,j] == 1 :
                cnt += 1
            else:
                if cnt >= 10:
                    ans += 1
                cnt = 0
    l.append(ans)
    return l

#feature 6
def cnt_vlines(img):
    r, c = img.shape
    ans = 0
    l = []
    for j in range(0, c):
        cnt = 0
        for i in range(0, r):
            if img[i, j] == 1:
                cnt += 1
            else:
                if cnt >= 10:
                    ans += 1
                cnt = 0
    l.append(ans)
    return l

def get_data(img):

    img = pre_init(img)
    features = []
    x,y=img.shape
    features+=x/y
    img = util.cnvt_bool_to_uint8(img)
    img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_CUBIC)
    
    curr = histo(img)
    features += curr

    img[img == 255] = 1

    curr = crossing(img)
    features += curr
    curr = zoing_new(img)
    features += curr
    curr = momentum(img)
    features += curr
    curr = cnt_hlines(img)
    features += curr
    curr = cnt_vlines(img)
    features += curr

    return features