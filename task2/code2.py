import cv2
import numpy as np
from sympy import N, pi
def pi_filter_fun(pi_img):
    distorted_digit=[]
    pi_filter_flat=[]

    total_digits=len(pi_img)
    pi_real = N(pi, total_digits+1) 
    original_pi=str(pi_real)
    original_pi=original_pi[:-1] # gives required digit by removing last (excluding rounded digits)
    original_pi=original_pi.replace(".","")

    for i in range(0,2500):
        if int(original_pi[i]) !=pi_img[i]/10:
            distorted_digit.append(int(original_pi[i])) #listing distorted digits

    for i in distorted_digit:
        pi_filter_flat.append(int(i*10*pi))

    pi_filter_flat =sorted(pi_filter_flat,reverse=True) #sorting distorted img in non-increasing order
    pi_filter=np.array(pi_filter_flat).reshape((2, 2))
    return pi_filter


pi_img=cv2.imread("pi_image.png", cv2.IMREAD_GRAYSCALE).flatten()
pi_filter=pi_filter_fun(pi_img)
print("2 X 2 matrix \n",pi_filter)

def recovered_image(distorted_path, pi_filter):
    img = cv2.imread(distorted_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    and_img= img.copy()
    or_img= img.copy()
    xor_img= img.copy()
    #applying different opperation to distorted image 
    for i in range(0, h-1, 2):
        for j in range(0, w-1, 2):
            and_img[i:i+2, j:j+2] &= pi_filter.astype(np.uint8) 
            or_img[i:i+2, j:j+2] |= pi_filter.astype(np.uint8)
            xor_img[i:i+2, j:j+2] ^= pi_filter.astype(np.uint8)
    
    cv2.imwrite("and_img.png", and_img)
    cv2.imwrite("or_img.png", or_img)
    cv2.imwrite("xor_img.png", xor_img)
    return and_img,xor_img,or_img


restored_image = recovered_image("distorted.png", pi_filter)[1]
print("shape of image restore img ",np.shape(restored_image))


def template_matching(collage, template):
    x=0
    y=0
    min_diff = float("inf")
    h, w = template.shape
    ch, cw = collage.shape
    
    for i in range(ch - h + 1):
        for j in range(cw - w + 1):
            diff = np.sum(collage[i:i+h, j:j+w] - template) # sum of difference btw two img
            if diff < min_diff:
                min_diff = diff
                x=i
                y=j
    return x,y
collage=cv2.imread("collage.png", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("gray_collage.png", collage)
x,y=template_matching(collage, restored_image)
print("coordinate of found template",x,y)
print("password",int((x + y) * pi)) 