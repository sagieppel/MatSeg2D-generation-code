import cv2
import numpy as np
import os
import UV_Map_Extractore as UV
import shutil
#####################################################################################################

# Generate image and its segmentation map (each segment is different texture)
# use natural image to generate segmentation and deploy one image as background and texture to each segment
# demand folder of uniform textures and folder of natural images
# Code should run as is with the sampled folder supplied

####################################################################################################

# tile image into grid (to fit sx,sy size)

###########################################################################3
def tile2grid(im,sx,sy):
    grid=im.copy()
    while(sx>grid.shape[1]):
        grid = np.concatenate([grid,im], axis=1)
    grid_layer=grid.copy()
    while (sy > grid.shape[0]):
        grid = np.concatenate([grid, grid_layer], axis=0)

    x0 = np.random.randint(grid.shape[1] - sx + 1)
    y0 = np.random.randint(grid.shape[0] - sy + 1)
    grid = grid[y0:y0 + sy, x0:x0 + sx]
    return grid
####################################################################################################

# resize image  (to fit sx,sy size)

###########################################################################3
def scale_texture(im,sx,sy):
    grid = im.copy()
    r = np.max([sx/im.shape[1],sx/im.shape[0]])
    sz = [np.max([int(r*im.shape[1]),sx]),np.max([int(r*im.shape[0]),sy])]
    grid = cv2.resize(grid,sz)
    x0 = np.random.randint(grid.shape[1] - sx + 1)
    y0 = np.random.randint(grid.shape[0] - sy + 1)
    grid = grid[y0:y0 + sy, x0:x0 + sx]
    return grid
######################################################################################################

# Load and augment texture from image of uniform texture

###############################################################################################
def load_texture(texture_list,sx,sy,min_tile=90):
    # --------Load Image---------------------------------------------
    while (True):
        ind = np.random.randint(len(texture_list))
        im = cv2.imread(texture_list[ind])
        if (im.shape[1] > sx and im.shape[0] > sy) or np.random.rand()<1: break
    if np.random.rand()<0.5: im = np.rot90(im)
    if np.random.rand()<0.5: im = np.fliplr(im)
    if np.random.rand()<0.5: im = np.flipud(im)
    #-----Tile to grid--------------------------------------------------------------------
    if np.random.rand()<0.75:
        # --------Resize----------------------------------------------------------------------------
        if np.random.rand() < 0.6:
            r = 1 + np.random.rand() * (np.min([im.shape[1] / min_tile, im.shape[0] / min_tile]) - 1)
            im = cv2.resize(im, [int(im.shape[1] / r), int(im.shape[0] / r)])
        im = tile2grid(im,sx,sy)
    else:
        im = scale_texture(im,sx,sy)
    if np.random.rand()<0.12: # Augment color
        print("augmenting color")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        im[:,:,0] = ((im[:,:,0].astype(np.float32)+np.random.randint(0,255))%255).astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
    return im
#######################################################################################################################

# Load image that will be used as background

#######################################################################################################################
def load_background(texture_list,sx=-1,sy=-1, resize=True, crop=True,display=False):
    # --------Load Image---------------------------------------------
    while (True):
        ind = np.random.randint(len(texture_list))
        im = cv2.imread(texture_list[ind])
        if (im.shape[1] > sx and im.shape[0] > sy): break
        if (im.shape[1] > sx and im.shape[0] > sy) or np.random.rand() < 0.03: break
    # --------if the loaded image is smaller then the target image resize
    r = np.min([im.shape[1] / sx, im.shape[0] / sy])
    if r <= 1:
        im = cv2.resize(im, (int(im.shape[1] / r) + 2, int(im.shape[0] / r) + 2))
    #--------Resize----------------------------------------------------------------------------
    if resize and sy>-1:
        if np.random.rand() < 0.5 and crop==True:
            r = 1 + np.random.rand() * (np.min([im.shape[1] / sx, im.shape[0] / sy]) - 1)
        else:
            r = np.min([im.shape[1] / sx, im.shape[0] / sy])
        im = cv2.resize(im, [1+int(im.shape[1] / r), 1+int(im.shape[0] / r)])
    #try:
    x0 = np.random.randint(im.shape[1] - sx + 1)
    y0 = np.random.randint(im.shape[0] - sy + 1)
    im = im[y0:y0 + sy, x0:x0 + sx]
    # except:
    #     cc=dfdf7

    if display: cv2.imshow("background",im);cv2.waitKey()
    return im
#
#####################################################################################################

# Generate image and its segmentation map (each segment is different texture)
# use natural image to generate segmentation and deploy one image as background and texture to each segment
# demand folder of uniform textures and folder of natural images

####################################################################################################
def generate_segmentation(image_dir,texture_dirs,out_dir,num_files,display=False):
# ----------Get Images and Texture List-----------------------------------------------------
    image_list = []
    for fl in os.listdir(image_dir):
        image_list.append(image_dir + "/" + fl)
    texture_list = []
    for texture_dir in texture_dirs:
        for fl in os.listdir(texture_dir):
            texture_list.append(texture_dir + "/" + fl)
 #####   if os.path.isdir(out_dir): shutil.rmtree(out_dir)
    if not os.path.isdir(out_dir): os.mkdir(out_dir)
# Start geneneate annotatio mmap
    for i in range(num_files):

        outsbdir = out_dir + "/" + str(i) + "/"
        print(outsbdir)
        if os.path.exists(outsbdir): continue
        if display: cv2.destroyAllWindows()

        # Set random parameters
        sx = np.random.randint(400,1000) # size
        sy = np.random.randint(400, 1000)
        num_mat = np.random.randint(1,11)
        hard_seg=np.random.rand()<0.55 # hard or soft segmentation
        max_occupancy = np.random.rand()*np.random.rand()*0.5+0.04 # max image accupancy for single sgment
        if np.random.rand()<0.1: max_occupancy=0.1
        # get segmentation map
        print("Creating UV map")
        print(" image size sx,sy ,num seg",sx,sy,num_mat)
        # Generate segmentation map
        uvmaps,background_map,color_segmentation = UV.get_uv_map(sx, sy, image_list, num_mat=num_mat, hard_seg=hard_seg,max_occupancy=max_occupancy)
        if display: cv2.imshow("uv map", color_segmentation);cv2.waitKey()
        # Add background to segmentation map
        background = load_background(image_list,sx=sx,sy=sy, resize=True, crop=True)
        if display: cv2.imshow("background",background);cv2.waitKey()
        img = background.astype(np.float32)*background_map[:,:,np.newaxis]
        if display: cv2.imshow("image background", img.astype(np.uint8));cv2.waitKey()
        # Set texture to each segment
        print("Add ",uvmaps.shape[0]," textures")
        # Add texture for each segment in the segmentation map
        for itx in range(uvmaps.shape[0]):
            if display: cv2.destroyAllWindows()
            if np.random.rand()<0.91:
                  print("loading texture")
                  texture=load_texture(texture_list, sx, sy, min_tile=100)
            else:
                 texture=np.zeros_like(img)
                 for ch in range(3): texture[:,:,ch]=np.random.randint(0,255)
            if display: cv2.imshow("texture" + str(itx), texture.astype(np.uint8));cv2.waitKey()
            if np.random.rand()<0.75: # Add shadow like effect
                 print("adding shadows")
                 texture = UV.add_shadows(texture,image_list,display)
                 if display: cv2.imshow("texture"+str(itx), texture.astype(np.uint8));cv2.waitKey()
            print("adding to uv map")
            img+=texture.astype(np.float32)*uvmaps[itx][:,:,np.newaxis]
            if display: cv2.imshow("img" + str(itx), img.astype(np.uint8));cv2.waitKey()
        print("Saving")
        # save image and annotation map to file
        img = img.astype(np.uint8)

        os.mkdir(outsbdir)
        cv2.imwrite(outsbdir+"/RGB__RGB.jpg",img.astype(np.uint8))
        cv2.imwrite(outsbdir + "/color_segmentation.jpg", color_segmentation)
        #cv2.imwrite(outsbdir + "/ROI.png", ((1 - background_map) * 255).astype(np.uint8))
        cv2.imwrite(outsbdir + "/ObjectMaskOcluded.png", ((1 - background_map) * 255).astype(np.uint8))
        cv2.imwrite(outsbdir + "/background.png", ((background_map) * 255).astype(np.uint8))
        for itx in range(uvmaps.shape[0]):
            cv2.imwrite(outsbdir + "/mask"+str(itx)+".png", ((uvmaps[itx]) * 255).astype(np.uint8))
        x=open(outsbdir+"/Finished.txt","w")
        x.close()
        print("finished:"+outsbdir)
#####################################################################################################

# Generate image and its segmentation map (each segment is different texture)
# use natural image to generate segmentation and deploy one image as background and texture to each segment
# demand folder of uniform textures and folder of natural images

######################################################################################################################################
if __name__ == '__main__':
    image_dir = "images/" # input images dir wil be used for both background and extraction of maps (see exampled dir)
    texture_dirs = ["textures/"] # input folder with various of textures (see example dir)
    out_dir = "2D_MateSeg/" # output dirs were the MatSeg images and their annotation will be saved
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    num_files = 10
    generate_segmentation(image_dir,texture_dirs,out_dir,num_files,display=False)

