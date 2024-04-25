# BGGAN: A Bidirectional Feature Generation Model for both Medical Images and Diagnostic Reports
---
## Requirements
- python 3.9
- Pytorch 1.12.1
- ## Installation
- Clone this repo.
```
git clone https://github.com/songlin-Chang/BGGAN
pip install -r requirements.txt
cd BGGAN/code/
```
## Preparation
### Datasets
1. Download the [IU X-ray](https://drive.google.com/file/d/1BUkGr0nWBmqvTV8bnYBzlKKuQ4HVY-IR/view?usp=drive_link) image data. Extract them to `data/xray/`
2. Download [MIMIC](https://drive.google.com/file/d/1XriBJAHJxVYIQiIXFYMblDgltUj2UAs9/view?usp=drive_link) dataset and extract the images to `data/mimic/`
## Evaluation
### Evaluate BGGAN T2I
We synthesize medical images from the test descriptions and evaluate the FID between **synthesized images** and **test images** of each dataset.
- For xray dataset(xray.yaml): `python calc_fid.py`
- For coco dataset(mimic.yaml): `python calc_fid.py`
- We compute inception score for models using [StackGAN-inception-model](https://github.com/hanzhanggit/StackGAN-inception-model).
### Evaluate BGGAN I2T
We synthesize diagnostic reports from the medical images and evaluate **synthesized diagnostic reports** and **test diagnostic reports** of each dataset.
- For xray dataset(config_xray.json): `python inference_test.py`
- For coco dataset(config_mimic.json): `python inference_test.py`
- using cocoevalcap to evaluate diagnostic reports(https://github.com/tylin/coco-caption/tree/master)
## Sampling
### Synthesize medical images
  ```
  python sample.py
  ```
### Synthesize diagnostic reports
  ```
  python single_test.py
  ```
