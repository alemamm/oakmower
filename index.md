## Depth smoothing tests

Weighted least squares filtering algorithm applied on OAK-D disparity stream:

![WLS filtering applied to indoor disparity](wls_filtering_disparity_indoor.png)

### Anomaly detection for segmented plane classifier

By intuition and observed behaviour of the plane segmentation while testing it indoors as well as outdoors I could assume that the parameters defining the plane in 3D should be represented by a Gaussian distribution (provided that the device is stable as well). Only if the RANSAC algorithm is not converging to similar results due to no significant portion of the point cloud containing a plane significant deviations are expected. Similarly, if no plane is detected, the amount of points contained in the algorithm's "best guess" plane will be low.

This allowed me to try and evalute anomaly detection algorithms visually keeping the amount of points in the plane on the x-axes as an indicator of how much I can rely on that data point. The y-dimension in the plots represents the a, b, c and d parameters describing the orientation of the plane, respectively.


Anomaly detection algorithms evaluated using unscaled default parameters:

![Anomaly detection algorithms evaluated using unscaled default parameters](anomaly_detection_defaults.png)


Anomaly detection algorithms evaluated using standard scaling and default parameters

![Anomaly detection algorithms evaluated using standard scaling and default parameters](anomaly_detection_stdscaler.png)


](anomaly_detection_stdscaler_outlierfrac_0.15.png)

![Anomaly detection algorithms evaluated using standard scaling and outlier fraction 0.15](anomaly_detection_stdscaler_outlierfrac_0.15.png)
