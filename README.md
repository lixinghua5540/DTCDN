# DTCDN
Title: *A deep translation (GAN) based change detection network for optical and SAR remote sensing images* [[paper]](https://www.sciencedirect.com/science/article/pii/S0924271621001842)<br>

X. Li, Z. Du, Y. Huang, and Z. Tan, “A deep translation (GAN) based change detection network for optical and SAR remote sensing images,” ISPRS Journal of Photogrammetry and Remote Sensing, vol. 179, pp. 14-34, September, 2021.
<br>
<br>
***Introduction***<br>
<br>
With the development of space-based imaging technology, a larger and larger number of images with different  modalities and resolutions are available. The optical images reflect the abundant spectral information and  geometric shape of ground objects, whose qualities are degraded easily in poor atmospheric conditions. Although synthetic aperture radar (SAR) images cannot provide the spectral features of the region of interest (ROI), they can capture all-weather and all-time polarization information. In nature, optical and SAR images encapsulate lots of complementary information, which is of great significance for change detection (CD) in poor weather situations.However, due to the difference in imaging mechanisms of optical and SAR images, it is difficult to conduct their CD directly using the traditional difference or ratio algorithms. Most recent CD methods bring image translation to reduce their difference, but the results are obtained by ordinary algebraic methods and threshold  segmentation with limited accuracy. Towards this end, this work proposes a deep translation based change detection network (DTCDN) for optical and SAR images. The deep translation firstly maps images from one domain (e.g., optical) to another domain (e.g., SAR) through a cyclic structure into the same feature space. With the similar characteristics after deep translation, they become comparable. Different from most previous researches, the translation results are imported to a supervised CD network that utilizes deep context features to separate the unchanged pixels and changed pixels. In the experiments, the proposed DTCDN was tested on four representative data sets from Gloucester, California, and Shuguang village. Compared with state-of-the-art methods, the effectiveness and robustness of the proposed method were confirmed. <br>
<br>
![Fig](https://user-images.githubusercontent.com/75232301/187924458-19ba5dd3-f4bc-4f25-958e-e5a7a349bd9e.jpg)<br>
<br>
<br>
***Usage***<br>
The implementation code of the proposed method consists of two parts:***Deep translation*** and ***Change detection*** <br>
First, you should run ***Deep translation*** folder. deep translation is the code of deep migration, and the input data need to cut the image into small pictures to build samples<br>
Second, ***Change detection*** floder is to use the migrated image for Change detection. The example data given here is Gloucester-SAR, but without data enhancement<br>


