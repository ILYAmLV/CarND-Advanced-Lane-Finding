
# Self-Driving Car Engineer Nanodegree

## Advanced Lane Finding - Project 4

---


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


### My project includes the following files:
* **P4.ipynb** containing the script to run the model.
* Three '.mp4' video files containing the result of this model.
* **'TEST_IMAGES'**  folder containing 6 original test images and 11 additional test images sourced from challenge_video and harder_challenge_video.
* **'OUTPUT_IMAGES'** folder containing visual respresentaions of steps that I took in creating this model
* **'WRITEUP.md'** markdown report which you are now reading.


### Camera Calibration 
---

The code for this step is contained in the third code cell of the IPython notebook **P4.ipynb**.

I have implemented a camera calibration class created by a fellow udacity student [mxbi](https://github.com/mxbi/advanced-lane-line-detection). The class creates a camera and distortion matrix that can be later used by calling the **`calibration.undistort`** or **`transform.warp`** functions. The class uses OpenCV's `findChessboardCorners` function to generate a set of image points to then create a matrix of (x, y, z) "object points".  

Then using the output `objpoints` and `imgpoints` the class computes the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  These variables are then stored inside the class until  distortion correction to the test image using the **`calibration.undistort`**:


### Pipeline (Thresholding)
---

To come up with an optimal thrsesholding combination for detecting yellow and white traffic lanes I have explored many different color spaces (contained in the `color_explr.ipynb` file), mainly the **HLS, LUV and LAB** color spaces:

I took the **`Saturation Channel`** from the **HLS** color space and thresholded it to pick up pixels only in the **170** - **255** range.

I took the **`Luminance Channel`** from the **LUV** color space and thresholded it to pick up pixels only in the **220** - **255** range.

I took the **`A-Channel`** from the **LAB** color space and thresholded it to pick up pixels only in the **95** - **155** range.

I came up with the thresholding ranges for these color spaces by separating the outputs and manually changing the values to see which ranges worked best.


I then combined the thresholded image with **`Sobel`** output to get the final image:


I have also tried using  **`contrast limited adaptive histogram equalization (CLAHE)`**, it is great for improving the contrast of the image especially in the low light environments but without an algorithm for recognizing image condition constant **`CLAHE`** filtering was adding too much noise.


### Pipeline (Persective Transform)
---

For the perspective transform I have implemented a class created by a fellow udacity student [mxbi](https://github.com/mxbi/advanced-lane-line-detection).

The code in this **`PerspectiveTransform`** class includes a function called **`warp` and `unwarp`** which is created by using **`cv2's getPerspectiveTransform and warpPerspective`** functions

```
class PerspectiveTransformer():
    def __init__(self, src, dist):
        self.Mpersp = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        
    # Apply perspective transform
    def warp(self, img):
        return cv2.warpPerspective(img, self.Mpersp, (img.shape[1], img.shape[0]))
    
    # Reverse perspective transform
    def unwarp(self, img):
        return cv2.warpPerspective(img, self.Minv, (img.shape[1], 
img.shape[0]))
```

The **`PerspectiveTransform`** class takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.array([[585, 460], [203, 720], [1127, 720], [695,460]]).astype(np.float32) 

dst = np.array([[320, 0], [320, 720], [960, 720],
[960, 0]]).astype(np.float32)

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.


### Pipeline (Lane Fitting)
---

For this part I used the code from [Udacity's "Advanced Lane Finding lesson"](https://classroom.udacity.com)

First I take a histogram of the **`Combined Binary Image`**:
```
histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
```
Set the parameters for the **Sliding Window** method, and plot the lanes onto a blank image:
```
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
```
Then **`unwarp`** the image and combine the results:
```
newwarp = transform.unwarp(color_warp)
result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
```
To calculate calculated the radius of curvature of the lane and the position of the vehicle with respect to the center I used the formula:

Represented by folowing code:
```
ym_per_pix = 30/720.0 # meters per pixel in y dimension
xm_per_pix = 3.7/700.0 # meters per pixel in x dimension
        

        left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])

        right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

        average_curve_rad = (left_curverad + right_curverad)/2
        curvature = "Radius of curvature: %.2f m" % average_curve_rad

        left_pos = (left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2])
        right_pos = (right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2])
        lanes_mid = (left_pos+right_pos)/2.0
        
        
        distance_from_mid = binary_warped.shape[1]/2.0 - lanes_mid
        
        mid_dist_m = xm_per_pix*distance_from_mid
```

Here is an example of my result on a test image:


### Pipeline (Video)
---

Here's a [link](./project_video_out.mp4) to my **`project_video`** result.

### Discussion
---

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. I got a decent result on my `project_video` with a fairly accurate lane plotting but on the `challenge_video` and the `harder_challenge_video` the lane plotting started off well and quickly got distorted and confused by the different lighting and shadows. The pipeline could be greatly improved with an implementation of a condition recognizing algorithm that would switch between the thresholding and filtering methods based on the condition of the image, also a different and a more robust algorithm for the `area of interest` selection would greatly improve the prediction of the plotting especially on the `harder_challenge_video` due to its many curves and turns. 

### Second submission
---

I experimented a bit with thresholding different color spectrums, particularly the HSV and the HSL spectrums:

```
HSV = cv2.cvtColor(your_image, cv2.COLOR_RGB2HSV)

#For yellow
yellow = cv2.inRange(HSV, (20, 100, 100), (50, 255, 255))

#For white
sensitivity_1 = 68
white = cv2.inRange(HSV, (0,0,255-sensitivity_1), (255,20,255))

sensitivity_2 = 60
HSL = cv2.cvtColor(your_image, cv2.COLOR_RGB2HLS)
white_2 = cv2.inRange(HSL, (0,255-sensitivity_2,0), (255,255,sensitivity_2))
white_3 = cv2.inRange(your_image, (200,200,200), (255,255,255))

bit_layer = your_bit_layer | yellow | white | white_2 | white_3
```

and it did not work out for me so I kept the original settings. I also tried implementing `cv2's` **`matchShapes`** function but ran into some complications and considering how late this project already is I decided to keep the original code.
