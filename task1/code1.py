import cv2
import numpy as np
import matplotlib.pyplot as plt
img_l=cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)#read img in gray scale
img_r=cv2.imread('right.png', cv2.IMREAD_GRAYSCALE)
height, width = img_l.shape
disparity_map = np.zeros((height, width), dtype=np.float32) #initial disparity map 
block_size=15
#remove border
for y in range(block_size, height - block_size):
    for x in range(block_size, width - block_size):

        best_offset = 0
        min_ssd = float('inf')#any ssd is less than infinity 
        patch_l = img_l[y - block_size:y + block_size + 1, x - block_size:x + block_size + 1] #patch in left img
        for offset in range(48):
            if x - offset - block_size < 0:  
                break
            patch_r = img_r[y - block_size:y + block_size + 1, x - offset - block_size:x - offset + block_size + 1] #patch in right img
            ssd = np.sum((patch_l - patch_r) ** 2)
            if ssd < min_ssd:
                min_ssd = ssd
                best_offset = offset 
        disparity_map[y, x] = best_offset # disparity of the perticylar pixel
    print("row no :",y)
depth_map = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) # scaling disparity from 0 to 255
depth_map = np.uint8(depth_map)

depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
cv2.imwrite('output_img.png', depth_colormap)
plt.imshow(cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Depth Map")
plt.show()