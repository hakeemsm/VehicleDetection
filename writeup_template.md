## Writeup Template

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply an optional color transform and append binned color features, as well as histograms of color to the hog feature vector 
* These two steps are performed after normalizing the features and randomizing the data for training and testing.
* Implement a sliding-window technique and use the trained SVC classifier to search for vehicles in images.
* Run the pipeline on a video stream (first on test_video.mp4 and later implemented on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/hog_viz1.png
[image3]: ./output_images/hog_viz2.png
[image4]: ./output_images/hog_color_viz1.png
[image5]: ./output_images/hog_color_viz2.png
[image6]: ./output_images/boxed_img.png
[image7]: ./output_images/sliding_windows.png
[image8]: ./output_images/heatmap_viz_cars.png
[image9]: ./output_images/heatmap_viz_nocars.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

The class `VehicleDetector` is the core component of this project. It contains methods to
* Extract spatial & histogram features
* Extract hog features
* Classify with linear SVC
* Calculate sliding window points
* Search for cars in windows
* Detect for multiples & false positives
* Visualize heatmap     

The constructor of the class takes in the paths to cars and not cars images to be trained on and the points to be used to compute sliding window axes

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

HOG features were extracted first by themselves using the method `extract_hog_features`.  This method takes in a struct to set all the parameters to be passed to hog as well as the image.

Cell 11 in the [notebook](./VehicleDetection.ipynb) contains the test code for extracting HOG features as well as classifying and visualizing them

The project starts off by extracting color hist features and classifying them by calling the method `color_classify_visualize` in the `VehicleDetector` class 

Here is an example of one of each of the `car` and `non-car` classes as well as their classification results using Linear SVC classifier

![alt text][image1]

The above was calculated using the parameters `color_space=RGB`, `spatial_size=(32,32)`, `hist_bins=32` and `hist_range=(0,256)` 

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`), running the classifier with each combination and measuring the performance for a good match

The image below is an example of computing just the hog features for all classes and running the classifier for that data set. The parameters for hog were `orient=7, pix_per_cell=(14, 14, cell_per_block=(2,2), hog_channel='ALL'`

![alt text][image2]

And here is an example using the HOG parameters of `orientations=32`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2), hog_channel='ALL'`:


![alt text][image3]

Further more, I explored with a combination of color hist & hog features. Here are the classification results and visualization for a combination with `orient = 7, pix_per_cell = (14,14), cell_per_block = (2,2) hog_channel='ALL'; color_space='RGB', spatial_size=(24,24), hist_bins=24, hist_range=(0,256)`

![alt text][image4]

And another combination using the values `color_space='YCrCb', spatial_size=(32,32), hist_bins=16, hist_range=(0,256); orient=32,pix_per_cell=(16,16), cell_per_block=(2,2), hog_channel='ALL' `

![alt text][image5]

#### 2. Explain how you settled on your final choice of HOG parameters.

I explored a range of options that yielded good results and which also had good performance. I intially started off with a smaller orientation and pix per cell values. While the classifier seems to perform alright on the training data, it seems to be overfitting. I then started varying the orientation, pix per cell and spatial size values while keeping the cell per block to 2x2 and a HOG channel value of `ALL` to extract feature for all channels. 

The combination that had the best performance for the classifier was the one with values:

`color_space='YCrCb', spatial_size=(32,32), hist_bins=16, hist_range=(0,256); orient=32,pix_per_cell=(16,16), cell_per_block=(2,2), hog_channel='ALL' `


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The classifier training is performed in the method `hog_color_classify_visualize` of the `VehicleDetector` class. This method uses both color & hog features using the values passed in as parameters. It splits the data into train & test sets with an 80/20 ratio and trains the classifier using the train data for both cars and not cars. It then measures the accuracy using the test data by calling the `score` method of the 
`SVC` classifier

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in the `extract_hot_windows` method of the `VehicleDetector` class. This method loops thru the coordinates provided to the class in the ctor and for each `xy_windows, xy_overlaps, xy_start_stops` it calls the method `slide_window` to get a collection of windows. These windows are then fed to `search_windows` method which loops thru the window list and for each window it calls the `single_img_feature` method. This method computes color & hog features for the passed in image which are used by the `search_windows` method to predict for presence of a car in the image by calling `predict` method on the SVC classifier. 

The points used for search were as below:

`xy_windows = [(64,64),(64, 64),(96, 96),(76, 76)],
xy_overlaps = [0.50, 0.50, 0.75, 0.80],
x_start_stops = [[575,1280], [640, 1280],[760, 1280],[760, 1280]],
y_start_stops = [[490, 720], [375, 720],[385,720],[395,720]]`

The xy_windows are the values for the boxed window dimensions and these were set based on whether a car is ahead in the same lane or in a different lane to the right or left and to also account for two different cars being side by side in a frame

The xy_overlaps are overlap values and these values have been set to account for all cars from left to right in a given frame. Having higher values in the beginning was causing some cars to be missed out while having smaller values in the end was causing cars in the right lanes to be only partially covered

The x_start_stops values were set to take into consideration the entire frame end-end so as not to leave out any lanes and also given the fact that the car being driven could be in any lane

The y_start_stops points were determined based on how much horizon coverage is needed for the frame



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I tried a combination of color space and hog values to optimize the classifier. All the data provided from the `vehicles` and `non-vehicles` datasets were used to train the classifier instead of using a subsample. This did result in longer classification times though

Here are some example images:

![alt text][image6]

![alt text][image7]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The method `add_heat` in conjunction with `apply_threshold` from the `VehicleDetector` class were used to apply filters, detect false positives and apply a heat map to the frames. The `extract_hot_windows` method used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are the frames, their corresponding heatmaps along with the `label` outputs for images with & without cars:

![alt text][image8]

![alt text][image9]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest issue I encountered is the time taken by the model to process the images. When I reduced the number of overlaps and start\stops, it was processing faster but seemed to miss out on some frames. Also the code seems to have been much faster with individual functions than a modular class like the one I coded.

Since the project video only has the car driving in the left lane, the model has not been tested for scenarios where the car would be in other lanes and might fail. Training the model with images of the vehicle being at different positions in the images might make it more robust  

