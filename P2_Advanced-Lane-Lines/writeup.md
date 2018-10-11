
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # 所有函数和算法都是使用./test_images下的图片进行测试的，所有图片均未输出到文件，仅可在ipynb文件中执行时查看。

### Camera Calibration

#### 1. 首先在课程中理解摄像机标定的原理，主要目的是要确认这个摄像头的畸变尺度，由于黑白棋盘本身的特点（都是直线，且每个点之间的距离相等），通过拍摄照片与已知的棋盘布局比对，就能修正照片的畸变，这是后续所有图像处理的基础。

### Pipeline (single images)

#### 1. 使用摄像头标定后的畸变修正矩阵，将照片修正（图片见project(2).ipynb文本）

#### 2. 识别车道线，在课程中我们学过了x轴梯度检测，y轴梯度检测，二维梯度检测，梯度方向检测以及颜色阈值检测等多种方法。本程序在经过反复测试后，最终选择x轴梯度、二维梯度以及HLS中的S阈值检测结合的方式，得到边缘检测的结果图（图片见project(2).ipynb文本）

#### 3. 选择ROI区域，一般选用平行四边形，这里要注意多留点空间，避免前方车道在转弯的过程中会超出ROI区域。（图片见project(2).ipynb文本）

#### 4. 对已经完成边缘能检测的灰度图像做透视变换，做法是用图片查看器放大直线车道图片，分别找到两条车道线同侧边缘的两个点，变换后即是鸟瞰图。（图片见project(2).ipynb文本）这里有一个坑，就是对于可见车道长度的判定。因为在后面计算曲率半径时需要将像素点转换成米，而这里做鸟瞰图的时候，鸟瞰图中车道线的显示长度是我指定的，指定不同的位置可能会导致不同的曲率，故后续不能单纯使用ym_per_pix = 30/720来转换像素和实际距离。

#### 5. 计算曲率半径，完全基于前面车道线识别的优劣。

#### 6. 最后就是上图，根据课程中的代码，将绿色填充到当前车道线中，以及自己网上查了一些资料，实现将文字输出到图像上。（图片见project(2).ipynb文本）

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

### Discussion

#### 1. 代码性能是一个很大的问题，先说结论，我用pipeline函数处理1260帧的项目视频时，总共耗时50多分钟，平均每帧2.5秒多。主要原因可能是没有按照课程中建议的简化滑动窗口检测的方法（没做过所以不确定能提升多少时间），主要是我还没有理解如何处理前后多张image的方法。

#### 2. 车道线检测做了好多好多测试，但是总是难以将阈值调节到适合所有照片的数值，这也是本次项目的核心，这个应该还是得依靠神经网络来调了。

#### 3. 曲率半径的计算，项目中说了这段路前一部分是均匀的1000m左右的曲率半径，但是我在视频处理中这个半径一直在2000m左右，很奇怪通过调节ym_per_pix参数也无济于事。

#### 4. 这个项目中涉及的信息量确实很大，每个函数我都花了好多时间去研究，这个项目对P1而言也是一种补充，毕竟P1无法处理弯曲的车道线。但是其实这种处理方式我觉的还是有不恰当的地方，就是在车辆变道时，车道线一定会离开ROI区域，不知道这种场景下该如何实践。

#### 5. 在尝试困难视频时，发现在曲率半径较小的道路上，在做透视变换时，会导致部分车道线像素变换后的位置在图片的尺寸之外，不知道关于这个问题处理的最佳实践是啥。