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

**Second Model: ResNet50**

Input: Input: 101 -> resize to 128
ResNet50 encoder
Decoder: 3 *  conv2d (1, 1), activation 'relu', batchNormalization with upsampling 

Created 5 Fold stratified by salt coverage: 

Training:

Initial Training fold1:
- 100 epochs, bce + diceloss 

![Resnet50 Predictions](https://github.com/GavinClarke0/TGS-Salt_Detection/blob/master/readme_files/resNet50_100epochbce_fold1.png "ResNet50 Fold 1 Model Predictions")

- 50 epochs, lovasz loss
  - learning rate too high, failed to include reduce on plateau
  
 ![Resnet50 Predictions](https://github.com/GavinClarke0/TGS-Salt_Detection/blob/master/readme_files/resNet50_50epochlovaz_fold1.png "ResNet50 Fold 1 Model Predictions")
 
 - 50 epochs, lovasz loss 
  -Predictions
 
 ![Resnet50 Predictions](https://github.com/GavinClarke0/TGS-Salt_Detection/blob/master/readme_files/resnet50train1pred.png "ResNet50 Fold 1 Model Predictions")

fold2:
- 100 epochs, bce + diceloss

![Resnet50 Predictions](https://github.com/GavinClarke0/TGS-Salt_Detection/blob/master/readme_files/resNet50_100epochbce_fold2.png "ResNet50 Fold 2 Model Predictions")



