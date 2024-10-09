# MatSeg2D-generation-code
Generation code for the MatSeg 2D using patterns infusion.
Use this code to generate the MatSeg 2D dataset described in [Learning Zero-Shot Material States Segmentation, by
Implanting Natural Image Patterns in Synthetic Data](https://arxiv.org/pdf/2403.03309).

For the MatSeg dataset see [1](https://zenodo.org/records/11331618](https://sites.google.com/view/matseg) or [2](https://zenodo.org/records/11331618)

For running the code see: Generate_2D_Scene_And_Annotation.py main section.

The script should run as is with the sampled folders supplied.

Was run on standart i7 cpu with no special hardware.

The MatSeg 3D generation code available at [this url](https://github.com/sagieppel/MatSeg-Synthethic-Dataset-Generation-Script).


# What does this do
This code use set of random images in one folder and set of texture images in second  folder to generate 2d dataset of texture/materials segmentation.
Basically the code extract various of patterns/shapes from the image folder and use them to create soft map/masks it then choose one random image and few random textures and map the textures into the image using the patterns extracted earlier.
See: [Learning Zero-Shot Material States Segmentation, by
Implanting Natural Image Patterns in Synthetic Data](https://arxiv.org/pdf/2403.03309) for more details.

The sample folder contains few random images and textures that should allow the code to run as is.

For large set of free textures see the vastexture dataset at: (1)[https://zenodo.org/records/12629301],(2)[https://zenodo.org/records/12629301].
For images it I used the (Segment anything dataset)[https://segment-anything.com/dataset/index.html] and (open images dataset)[https://storage.googleapis.com/openimages/web/index.html].

