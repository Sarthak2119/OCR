import cv2

def line_seg(img):
    lines = []
    r, c = img.shape
    prev = 0
    for i in range(0,r):
        cnt = 0
        for j in range(0,c):
            if (img[i, j] == 255):
                cnt = cnt+1
        if (cnt == c) and (prev - i > 1):
            lines.append(img[prev:i+1, :])
            prev = i + 1
    return lines