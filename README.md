https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/

Install python packages
```
pip3 install albumentations tqdm
```

Make a folder for storing outputs
```
mkdir outputs
```

### opencv error
ptaranov@ptswt3:~/src/watches/clockworks/classificator/testsets/creatures$ Dataset.py training/
Number of training images: 30
Traceback (most recent call last):
  File "/home/ptaranov/bin/Dataset.py", line 204, in <module>
    visualize_sample(image, target)
  File "/home/ptaranov/bin/Dataset.py", line 199, in visualize_sample
    cv2.imshow('Image', image)
cv2.error: OpenCV(4.9.0) /io/opencv/modules/highgui/src/window.cpp:1272: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'

```
pip uninstall opencv-python; pip install opencv-python
```
