import os
import cv2
import shutil
import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu
import scipy
from sklearn.metrics.pairwise import cosine_similarity
import json
#########################################################################################################################33

# Generate segmentation map from natural image in an unsupervised

####################################################################################################################3

def probability_map_to_one_hot(prob_map):
    """
    Convert a probability map to a one-hot encoded map.

    Parameters:
    - prob_map: A numpy array of shape [c, h, w] containing probability values.

    Returns:
    - A numpy array of the same shape, where for each location (x, y),
      the class with the highest probability is marked as 1, and all others as 0.
    """
    # Find the index of the maximum probability class for each pixel
    max_prob_class_indices = np.argmax(prob_map, axis=0)

    # Create a one-hot encoded map
    one_hot_map = np.zeros_like(prob_map)
    c, h, w = prob_map.shape
    for x in range(h):
        for y in range(w):
            one_hot_map[max_prob_class_indices[x, y], x, y] = 1
    one_hot_map[prob_map<0.1]=0

    return one_hot_map
#

###########################################################################################################3

# Turn probability map to RGB for visualization

#############################################################################################################
def probability_map_to_rgb(prob_map,display=False):
    # Define a simple color palette for up to 20 classes, using RGB tuples.
    # Extend or modify this palette as needed for more classes or different colors.
    color_palette = np.array([
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (192, 192, 192),# Silver
        (128, 0, 0),    # Maroon
        (128, 128, 0),  # Olive
        (0, 128, 0),    # Dark Green
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (0, 0, 128),    # Navy
        (139, 69, 19),  # SaddleBrown
        (75, 0, 130),   # Indigo
        (255, 165, 0),  # Orange
        (255, 20, 147), # DeepPink
        (0, 100, 0),    # DarkGreen
        (0, 191, 255),  # DeepSkyBlue
        (210, 105, 30), # Chocolate
    ], dtype=np.float32)

    # Ensure the color palette has enough colors for the number of classes
    num_classes = prob_map.shape[0]
    if num_classes > color_palette.shape[0]:
        raise ValueError("Number of classes exceeds the size of the predefined color palette.")

    # Normalize the color palette to [0, 1] for averaging
    color_palette /= 255.0

    # Calculate the weighted average color for each pixel
    # Reshape prob_map from [c, h, w] to [h, w, c] for broadcasting with color_palette [c, 3]
    prob_map_transposed = np.transpose(prob_map, (1, 2, 0))
    # Multiply the probabilities by the color palette and sum along the classes axis to get the average color
    rgb_image = np.tensordot(prob_map_transposed, color_palette[:num_classes], axes=([2], [0]))

    # Ensure the resulting RGB values are within the [0, 1] range
    rgb_image = np.clip(rgb_image, 0, 1)
    if display:
        cv2.imshow("",rgb_image)
        cv2.waitKey()
    return (rgb_image*255).astype(np.uint8)
##################################################################################################

# Create artificial shadows by darkening some regions (extracted from natural image)

####################################################################################################
def add_shadows(texture,image_list,display=False):
    sy = texture.shape[0]
    sx = texture.shape[1]
    while (True):
        ind = np.random.randint(len(image_list))
        im = cv2.imread(image_list[ind])
        if (im.shape[1] > sx and im.shape[0] > sy) or np.random.rand() < 0.04: break
# --------if the loaded image is smaller then the target image resize
    r = np.min([im.shape[1] / sx, im.shape[0] / sy])
    if r <= 1:
        im = cv2.resize(im, (int(im.shape[1] / r) + 2, int(im.shape[0] / r) + 2))
 # -------Resize--------------------------------------------
    if np.random.rand() < 0.3:
        r = 1 + np.random.rand() * (np.min([im.shape[1] / sx, im.shape[0] / sy]) - 1)
        im = cv2.resize(im, [int(im.shape[1] / r), int(im.shape[0] / r)])

        # cv2.imshow("resize"+str(im.shape),im);cv2.waitKey()
# --------crop---------------------------------------------------

    x0 = np.random.randint(im.shape[1] - sx + 1)
    y0 = np.random.randint(im.shape[0] - sy + 1)
    im = im[y0:y0 + sy, x0:x0 + sx]

    # cv2.imshow("croped"+str(im.shape), im);cv2.waitKey()
    if len(im.shape) < 3:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    if np.random.rand() < 0.4:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    map=im[:,:,np.random.randint(3)].astype(np.float32)


    #norm_img = img / 255.0 # Normalize the image to the range 0-1

    shadows = (map / map.max())
    if display: cv2.imshow("shadows",(shadows*255).astype(np.uint8));cv2.waitKey()
    kernel_size= 3+ np.random.randint(1,10)*2
    shadows = cv2.GaussianBlur(shadows,(kernel_size,kernel_size),0)
    if display: cv2.imshow("shadows blur",(shadows*255).astype(np.uint8));cv2.waitKey()
    if np.random.rand()<0.5:
         r =  1
    else:
         r=np.random.rand()
    texture=texture.astype(np.float32)
    texture = texture*(1-r)+texture*r*shadows[:,:,np.newaxis]
    if display: cv2.imshow("shadows img", texture.astype(np.uint8));cv2.waitKey()
    return texture

######################################################################################################

# Turn image into binary segmentation map, by picking image property and thresholding it (both random picked)

######################################################################################################
def get_uv_binary_map(img,max_occupancy=1,hard_seg=False):
    # pick random image property
    if np.random.rand() < 0.3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    map=img[:,:,np.random.randint(3)].astype(np.float32)

    # normalize map value
    #norm_img = img / 255.0 # Normalize the image to the range 0-1
    if np.random.rand()<0.5:
        map = (map / map.max())
    else:
        map = (map / 255)
    # Pick high threshold
    itr = 0
    while(True):
        high_thresh = np.random.rand()
        upper=map>high_thresh
      ##  if np.random.rand()<0.01: continue
        itr+=1
        if itr>20: return np.zeros_like(map)
        upmn = upper.mean()
        if upmn > max_occupancy: continue
        upmn = np.min([upmn,1-upmn])
        upsm = np.min([1-upper.sum(),upper.sum()])
        #if t>0.9: continue
        #if mn>0.5 and np.random.rand()>(1-mn)/2: continue
       # cv2.imshow("upper", upper.astype(np.uint8) * 255);cv2.waitKey()
        if upmn>0.01 or upsm>2000: break



    # Pick low threshold
    while (True):
            itr += 1
            if itr > 30: return np.zeros_like(map)
            low_thresh = np.random.rand()*high_thresh
            if np.random.rand()<0.5 or hard_seg:
                low_thresh=high_thresh



            lower = map < low_thresh

         #   cv2.imshow("lower", lower.astype(np.uint8) * 255);cv2.waitKey()
            mid = (1 - lower) * (1 - upper) > 0
            if np.random.rand() < 0.01: continue

            lmn = lower.mean()
            lmn = np.min([lmn, 1 - lmn])
            lsm = np.min([1 - lower.sum(), lower.sum()])
            if lsm==0: continue
            if  ((mid.sum()/lsm)>0.33 or (mid.sum()/upsm)>0.33) and np.random.rand()>0.1: continue
            if lmn > 0.01 or lsm > 2000 or np.random.rand() > 0.1: break

           # #
           #  if t > 0.01 or upper.sum() > 1000 or np.random.rand() > 0.1: break
    # apply threshold to selected map
    output = np.zeros_like(lower,dtype=np.float32)
    output[mid] =  (map[mid] - low_thresh) / (high_thresh - low_thresh+0.0000001)
    output[upper] = 1
    output[lower] = 0
    # cv2.imshow("final", output.astype(np.uint8) * 255);
    # cv2.waitKey()


    return  output


#######################################################################################################################

# Use multiple natural image to generate segmentation map with natural shapes

#####################################################################################################################
def get_uv_map(sx ,sy ,image_list , num_mat,hard_seg,max_occupancy):
    uv_list = []
    for i in range(num_mat):
        while(True):
            #--------Load Image---------------------------------------------
            while(True):
                ind = np.random.randint(len(image_list))
                im = cv2.imread(image_list[ind])
                if (im.shape[1]>sx and  im.shape[0]>sy) or np.random.rand()<0.05: break
            #--------if the loaded image is smaller then the target image resize
            r=np.min([im.shape[1]/sx,im.shape[0]/sy])
            if r<=1:
                im = cv2.resize(im,(int(im.shape[1]/r)+2,int(im.shape[0]/r)+2))

            #-------Resize--------------------------------------------
            if np.random.rand()<0.3:
                r=1+np.random.rand()*(np.min([im.shape[1]/sx,im.shape[0]/sy])-1)
                im = cv2.resize(im, [int(im.shape[1]/r), int(im.shape[0]/r)])

            #cv2.imshow("resize"+str(im.shape),im);cv2.waitKey()
            #--------crop---------------------------------------------------

            x0 = np.random.randint(im.shape[1]-sx+1)
            y0 = np.random.randint(im.shape[0]-sy+1)
            im=im[y0:y0+sy,x0:x0+sx]

            #cv2.imshow("croped"+str(im.shape), im);cv2.waitKey()
            if len(im.shape)<3:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            uv_map=get_uv_binary_map(im,max_occupancy=max_occupancy,hard_seg=hard_seg)
            if uv_map.sum()>100: break
        #-----------------------------------------------------------------
        uv_list.append(uv_map)
    uvmaps=np.asarray(uv_list)
    if np.random.rand()<0.5: # blurry
        kernel_size = 3 + np.random.randint(0, 4) * 2
        for kk in range(uvmaps.shape[0]):
            uvmaps[kk] = cv2.GaussianBlur(uvmaps[kk], (kernel_size, kernel_size), 0)
    if np.random.rand()<0.25: # normalize to one even
       uvmaps = uvmaps / (uvmaps.sum(0) + 0.000001)
    else: # normalize to one prioritize firsst
       p=np.ones_like(uvmaps[0])
       for kk in range(uvmaps.shape[0]):
           uvmaps[kk]*=p
           p-=uvmaps[kk]
    if hard_seg: # turn into one hot if hard segmentation
        uvmaps=probability_map_to_one_hot(uvmaps)
    rgb = probability_map_to_rgb(uvmaps) # make into color map
    background_map = 1-uvmaps.sum(0)
    return uvmaps,background_map,rgb
####################################################################################################################
# if __name__ == '__main__':
#     image_dir = "/media/breakeroftime/2T/Data_zoo/dms_v1_labels/images/train/"
#     image_list = []
#     for fl in os.listdir(image_dir):
#            image_list.append(image_dir+"/"+fl)
#
#     while(True):
#         uvmaps,background_map,rgb = get_uv_map(sx=500, sy=500, image_list=image_list, num_mat=np.random.randint(11)+1, hard_seg=np.random.rand()<0.5,max_occupancy=0.1)
#         probability_map_to_rgb(uv_maps)


