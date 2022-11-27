# RobustNPC
*Deep learning-based accurate and robust delineation of primary gross tumor volume of nasopharyngeal carcinoma on heterogeneous magnetic resonance imaging: a largescale and multi-center study*

## Notes
* This work was modified from [nnUNet](https://github.com/MIC-DKFZ/nnUNet).
* Due to data privacy protection, we can not release the five hospital datasets, but we released the trained our proposed-framework for new data prediction.

## How to use
### 1. Before you can use this package for NPC segmentation. You should install:
* PyTorch version >=1.8
* Some common python packages such as Numpy, Pandas, SimpleITK,OpenCV, pyqt5, scipy......
### 2. Run the inference script.
* Download the trained model (trained on 600 T1-weight MRI images from 3 hospitals) from [Google Drive](https://drive.google.com/drive/folders/1gapzMiF5c_-lBhI02xXPCWfYY21A9hhy) to ``./pretrained_model/``.
* Now, you can use the following code to generate NPC-GTVp delineation.
```python
from InferRobustNPC import InferRobustNPC
Inference3D(rawf="example.nii.gz", save_path="example_pred.nii.gz")
```


