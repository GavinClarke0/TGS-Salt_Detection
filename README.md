# TGS-Salt_Detection-
Working solution To Kaggle competition "TGS Salt Identification Challenge "


**Initial Model: VGG16 UNet**

Input: Input: 101 -> resize to 128, increase channels to 3
VGG16 encoder: pretrained with "Imagenet" weights
Decoder: conv2d (3, 3), activation 'selu', upsampling 

***Local Validation***

Created singular Fold stratified by salt coverage: 

Initial Training:
- 50 epochs, binary cross entropy loss
- 50 epochs, bce + diceloss 

Test Data Score: 0.635534

**VGG16 Prediction Visualization**

![VGG16 Predictions](https://github.com/GavinClarke0/TGS-Salt_Detection/blob/master/readme_files/myplot.png?raw=true "VGG16 Model Predictions")

**Second Model: ResNet50 **

Input: Input: 101 -> resize to 128
ResNet50 encoder
Decoder: 3 *  conv2d (1, 1), activation 'relu', batchNormalization with upsampling 

Created 5 Fold stratified by salt coverage: 

Initial Training:

fold1:
- 100 epochs, bce + diceloss 






