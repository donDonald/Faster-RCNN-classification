# Intro
It's an example of object classificator made on top of [PyTorch](https://pytorch.org/) and [Faster-RCNN](https://www.geeksforgeeks.org/faster-r-cnn-ml/) for training custom classification model.\
Usefull for the cases if here is need to detect some classes which are not covered by existing models.




# Setup 
Install python package.\
1-st create local py environment.
```
python3 -m venv .venv \
 && source .venv/bin/activate
```
Starting now on, assume that any py code is run for that newly created python environment, i.e. ***source .venv/bin/activate*** as called.


2-nd install all andatory python packages:
```
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install opencv-python \
 && pip install opencv-contrib-python \
 && pip install albumentations tqdm matplotlib
```




# Training classification
Current solution has some test sets for testing how it works:
* testsets/creatures
* testsets/vegetables
* testsets/vehicles

***Classificator.py*** tool privided for training classification.\
To train a model against ***creatures*** test set:
```
./Classificator.py ./testsets/creatures/training/ ./testsets/creatures/validation/ ./testsets/creatures/out/ -v1
```

For more details:
```
./Classificator --help
```




# Visualizing a dataset
***Dataset.py*** tool privided for visualizing datasets.\
To visualize ***./testsets/vehicles/training/*** data set:
```
./Dataset.py ./testsets/vehicles/training/
```

<p align="center"><img width='50%' src="images/1.png"></p>

For more details:
```
./Dataset.py --help
```




# Referenses
* https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/
* https://pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
* https://www.geeksforgeeks.org/faster-r-cnn-ml/
* [OBJECT_DETECTION_YOLO_VS_FASTER_R-CNN.pdf](docs/OBJECT_DETECTION_YOLO_VS_FASTER_R-CNN.pdf)

