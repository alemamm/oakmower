# Spatial AI for Limit and Obstacle Detection

## Problem statement
Lawn mowing is a time-consuming and tiring routine household task. In recent years it has become one of the main personal robot applications. But even high-end products still require the expensive installation of boundary wires to ensure that the robot stays on the lawn. However, even with sich wires limiting the lawn area, numerous problems can occur, including the bot
* killing small animals such as hedgehogs,
* hurting children, cats, dogs,
* driving into molehills and
* crashing into "unwired" obstacles such as trees.

Proper abstacle detection for mowing bots can save lives, money and the bot itself. The aim is to prove that this can be achieved using the OpenCV AI Kit with Depth (OAK-D).

## Problem description
Considering the importance of health and safety and the dangers posed by a lawn mower's cutting blades, any solution for limit and obstacle detection needs to be sufficiently robust. The most common approach today are boundary wires that need to be burried surrounding and thus limiting the lawn area. Any other obstacle is detected with contact sensors once the robot crashes into it. However, the robot will require some time to actually stop. If the obstacle is small, light or flat enough, the robot will not stop. As the number of lawn mower bots increases so do the stories about cut cattails, dead hedgehogs, etc. Click on the video below for an illustration of our bot's behaviour when encountering obstacles.

[![Problem visualisation Youtube video](youtube_problem_visualization_600.png)](https://www.youtube.com/watch?v=kr37imhNvWI)

# Solution
Spatial AI allows for multimodal solutions. OAK-D makes Spatial AI and Embedded AI available for everyone. I tried to fully leverage the power and functionality of OAK-D using all sensors and functionalities simultaneously:
* Neural inference for object detection on Intel Movidius Myriad X and 4K RGB camera
* Point cloud classification based both mono cameras for disparity/depth streams
* Disparity image classification based on disparity and rectified right streams
* Motion estimation using ectified right stream

![OAKMower flowchart](oakmower_flow_chart.png)

OAKMower uses three classifers for limit and obstacle detection:
* Point Cloud ([Elliptic Envelope for Outlier Detection](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html))
* Disparity ([Support Vector Machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html))
* Objects ([Mobilenet-SSD for Object Detection](https://docs.luxonis.com/tutorials/openvino_model_zoo_pretrained_model/))

Since for now control is out of scope, optical flow estimation was added to the pipeline to classify the movement of the robot.

## Setup

## Point Cloud classification
#### Idea
The point cloud obtained if enough free space is left in front of the bot should contain a significant amount of points belonging to a plane. Parameters describing the position of that plane in space should be relatively constant, mainly influenced by uneven terrain and camera movement. More points contained in the plane should mean more free space and higher confidence.

#### Approach
The device does not allow for streaming both disparity and depth. Also, that would be a waste of bandwdith and frame rate as depth can be calculated from disparity as explained [in the DepthAI FAQs](https://docs.luxonis.com/faq/#how-do-i-calculate-depth-from-disparity). Using the intrinsics from the calibration files, [a point cloud can be calculated using Open3D.](https://github.com/luxonis/depthai-experiments/tree/master/point-cloud-projection) The [plane can be segmented surprisingly fast within few iterations of the RANSAC algorithm.](http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html#open3d.geometry.PointCloud.segment_plane) This allows for real-time use on the host.

I used the parameters a, b, c and d

![Plane parameters](https://wikimedia.org/api/rest_v1/media/math/render/svg/5e85b2d4c03909f8388d6424de28d27870977972)

describing the plane as well as the amount of points considered as inliers of the plane as features to classify the point cloud as representing a clear path or obstacle. We can assume that the parameters of big clear path planes follow a normal distribution whereas obstacles can be considered outliers. Therefore, I decided to go for [anomaly detection algorithms](https://scikit-learn.org/stable/modules/outlier_detection.html) for the point cloud classifier.

## Filtered Disparity classification
#### Idea
The disparity/depth data is noisy and largely influenced by oclusions. Even assuming that outlier filtering is applied successfully, it will not be possible to tell if a flat area in front belongs to the lawn or plaster. Texture should be considered as well. Images combining both informations should allow for successful classification.
#### Approach
OpenCV implements a tunable [Weighted Least Squares disparity filter](https://docs.opencv.org/3.4/d9/d51/classcv_1_1ximgproc_1_1DisparityWLSFilter.html) to refine the results in half-occlusions and uniform areas. Luxonis also provides a stand-alone [WLS filter example](https://github.com/luxonis/depthai-experiments/tree/master/wls-filter). The filtered image is expected to look different when comparing flat/lawn areas to limits or obstacles.
##### Issue: Minimum Depth
Points below minimum depth will partially be filtered out. The remaining "holes" are present and look similar qualitatively on all images whereas the texture changes significantly in case of limits or obstacles.
##### Issue: Noisy Absolute Values
Even the WLS-filtered disparity still contains noise. To get grayscale invariance as well as translational invaariance, [local binary patterns](https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.local_binary_pattern) were computed for the relevant image area and binned in a relative histogram serving as the input for a [Support Vector Machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) classifier.

###### Sidenote: Minimum Depth
The [minimum depth](https://docs.luxonis.com/faq/#onboard-camera-minimum-depths) of the device depends on the camera baseline. In the case of the [OAK-D](https://docs.luxonis.com/products/bw1098obc/) used in this project the minimum depth was 0.689 meters. In my tests WLS filtering did a good job enhancing the results without needing the [extended disparity mode](https://docs.luxonis.com/faq/#extended_disparity) that is planned to be implemented in future DepthAI releases.
###### Sidenote: Semantic Segmentation:
Training and running a semantic segmentation model on the device on RGB images is a promising approach. However, the texture can also be seen in the mono camera stream and until the release of Gen2 the DepthAI API does not allow for inference of two models.

This should give an overview of the solution. Use diagrams, flowcharts to explain the solution visually. If you reference papers, please provide links. 

# Results [3 - 4 pages]
Use as much space as you want to show as many results as you want. Link to generated videos if needed.  

# Limitations [ 1 -2 paragraphs ]
Conditions under which the solution does not work.

# Future work [0.5 - 1 page]
Things that you could not complete because of time, budget, or other constraints. 

# References [0.5  - 1 page ]
Acknowledge code you may have used from other repositories. Also, refer to papers you may have implemented. 
