# RobustNPC
[**Deep learning-based accurate delineation of primary gross tumor volume of nasopharyngeal carcinoma on heterogeneous magnetic resonance imaging: a large-scale and multi-center study**](https://www.sciencedirect.com/science/article/pii/S016781402300018X)

## Notes
* This work was modified from [nnUNet](https://github.com/MIC-DKFZ/nnUNet).
* Due to data privacy protection, we can not release the five hospital datasets, but we released the trained our proposed-framework for new data prediction.

## How to use
### 1. Before you can use this package for NPC segmentation. You should install:
* PyTorch version >=1.8
* Some common python packages such as Numpy, SimpleITK, OpenCV, Scipy......
### 2. Run the inference script.
* Download the trained model (trained on 600 T1-weighted MRI images from 3 hospitals) from [Google Drive](https://drive.google.com/drive/folders/1gapzMiF5c_-lBhI02xXPCWfYY21A9hhy) to ``./pretrained_model/``.
* Now, you can use the following code to generate NPC-GTVp delineation.
```python
from InferRobustNPC import Inference3D
Inference3D(rawf="example.nii.gz", save_path="example_pred.nii.gz") # rawf is the path of input image; save_path is the path of prediction.
```
* The trained model just can predict the T1-weighted MRI images, the thickness should be in the range of 2.5mm-10.0mm (<1.0mm images will be supported later). 

* This project was originally developed for our previous work [RobustNPC](https://www.sciencedirect.com/science/article/pii/S016781402300018X), if you find it's useful for your research, please consider to cite the followings:
    @article{luo2023deep,
    title={Deep learning-based accurate delineation of primary gross tumor volume of nasopharyngeal carcinoma on heterogeneous magnetic resonance imaging: a large-scale and multi-center study},
    author={Luo, Xiangde and Liao, Wenjun and He, Yuan and Tang, Fan and Wu, Mengwan and Shen, Yuanyuan and Huang, Hui and Song, Tao and Li, Kang and Zhang, Shichuan and Zhang, Shaoting and Wang, Guotai},
    journal={Radiotherapy and Oncology},
    pages={109480},
    year={2023},
    publisher={Elsevier}
    }


## Acknowledgment and Statement
If you have any question, please contact [Xiangde Luo](https://luoxd1996.github.io).


