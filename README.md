# MatSeg2D-generation-code
Generation code for the MatSeg 2D using patterns infusion.
Use this code to generate the MatSeg 2D dataset described in [Learning Zero-Shot Material States Segmentation, by
Implanting Natural Image Patterns in Synthetic Data](https://arxiv.org/pdf/2403.03309).

For the MatSeg dataset see [1](https://zenodo.org/records/11331618] or [2](https://sites.google.com/view/matseg)

For running the code see: Generate_2D_Scene_And_Annotation.py main section.

The script should run as is with the sampled folders supplied.

The code was run on a standard i7 CPU with no special hardware.

The MatSeg 3D generation code is available at [this url](https://github.com/sagieppel/MatSeg-Synthethic-Dataset-Generation-Script).

![Results images and annotation](/Example_results.jpg)



# What does this do
This code uses a set of random images in one folder and a set of texture images in the second folder to generate a 2D dataset of texture/materials segmentation.
Basically, the code extracts various patterns/shapes from the image folder and uses them to create soft maps/masks. It then chooses one random image and a few random textures and maps the textures into the image using the patterns extracted earlier. The result is random images with various textures scattered on it and the segmentation map. This can be used to train the net for material state segmentation.
See: [Learning Zero-Shot Material States Segmentation, by
Implanting Natural Image Patterns in Synthetic Data](https://arxiv.org/pdf/2403.03309) for more details.

The sample folder contains a few random images and textures that should allow the code to run as is.

For a large set of free textures see the vastexture dataset at: (1)[https://zenodo.org/records/12629301],(2)[https://zenodo.org/records/12629301].

For a large set of free images  (Segment anything dataset)[https://segment-anything.com/dataset/index.html] and (open images dataset)[https://storage.googleapis.com/openimages/web/index.html].
![Data Generation Scheme](/Scheme_Small.jpg)
