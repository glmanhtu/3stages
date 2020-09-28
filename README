
# Three stages training for pain estimation using PyTorch  
  
This is a PyTorch implementation of the pain estimator described in the paper ["Automated Pain Estimation based on FacialAction Units from Multi-Databases"]("")

## Compatibility
The code is tested using PyTorch 1.6 and Torchvision 0.7 under Ubuntu 16.04.6 LTS with Python 3.7.  The dependencies can be installed by running the following command:
```console
pip install -r requirements.txt
```
## Pre-trained models
All the pre-trained models for each of the subjects in the UNBC McMaster database are stored in Google Drive and will be automatically downloaded when run the demo or validation. The IDs of those pre-trained model can be found in /resources/unbc/pretrained_models.json

## Testing
There are two scripts for testing. *demo_cnn.py* for testing the first two stages mentioned in the paper, and *demo_cnn_lstm.py* for testing the all three stages.
```console
usage: demo_cnn.py [-h] [--webcam] [--video-file VIDEO_PATH]

Pain level intensity estimation using Stage 1+2

optional arguments:
  -h, --help            show this help message and exit
  --webcam              	Using webcam as input
  --video-file VIDEO_PATH	Path to video file
```

## Validating
For data preparation, please make a request for the [UNBC McMaster database](https://www.pitt.edu/~emotion/um-spread.htm), extract and put it under /resources/unbc/dataset folder. As we are using the pre-trained models, there are no need to download the DISFA dataset.

Run the following command to perform the leave-one-subject-out cross-validation:
```console
python validation.py
```
The pre-trained models will be downloaded automatically and the validation results can be seen in the console output.

## Citation

