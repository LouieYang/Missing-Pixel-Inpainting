# Missing-Pixel-Inpainting
Implemented color and grayscale missing pixel inpainting paper

# QUESTION
The original image is masked like Figure 1, where the mask's pixel value is zero. If the image is RGB, then mask each channel independently. **The task is to inpaint the pixel of 0 on the masked image**.
<div align="center">
  <img src="https://github.com/LouieYang/Missing-Pixel-Inpainting/blob/master/example/A.png">
  <img src="https://github.com/LouieYang/Missing-Pixel-Inpainting/blob/master/example/A_ori.png"> <br/>
  Figure 1. grayscale 80% masked image
</div>
<div align="center">
  <img src="https://github.com/LouieYang/Missing-Pixel-Inpainting/blob/master/example/B.png">
  <img src="https://github.com/LouieYang/Missing-Pixel-Inpainting/blob/master/example/B_ori.png"> <br/>
  Figure 2. color 40% masked image <br/> <br/>
</div>
<div align="center">
  <img src="https://github.com/LouieYang/Missing-Pixel-Inpainting/blob/master/example/C.png">
  <img src="https://github.com/LouieYang/Missing-Pixel-Inpainting/blob/master/example/C_ori.png"> <br/>
  Figure 3. color 60% masked image <br/> <br/>
</div>

# RESULT
<div align="center">
  <img src="https://github.com/LouieYang/Missing-Pixel-Inpainting/blob/master/example/A_inpainted.png">
  <img src="https://github.com/LouieYang/Missing-Pixel-Inpainting/blob/master/example/B_inpainted.png">
  <img src="https://github.com/LouieYang/Missing-Pixel-Inpainting/blob/master/example/C_inpainted.png">
  Figure 3. Inpainting CNN Result
</div>
<br/>
<br/>


**Table 1. L2 Norm per missing pixel of example**

| **TYPE** | **80% Grayscale** | **40% Color** | **60% Color** |
|--------------------------|---------------------|------------------|-------------------| 
|L2 Norm per missing pixel | 0.10093 |0.03512 | 0.08130|

# Training
The **preprocessing.py** will generate four .npy files, like train_x_color.npy, train_y_color.npy, train_x_gray.npy, train_y_gray.npy.

```
python train.py train_gray 1 (train gray model and will generate ./gray-model/
python train.py train_gray 0 (train color model and generate ./color-model/)
```
The pre-processed data can be downloaded on [preprocessed data](https://pan.baidu.com/s/1eS03UgE) and pre-trained model could be download on [model](https://pan.baidu.com/s/1kUW74Pp), where the dataset is randomly select from MSCOCO
# USAGE
```
python inpainting --img_dir some-of-missing-pixel-image --model_version XXXX (where XXXX is the best trained model)
```

# REFERENCE
[1] [https://arxiv.org/pdf/1611.04481.pdf](paper)
