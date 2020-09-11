# Task 2 - Gender Classification

A model trained to classify a person's gender by a photo.

---

## How to use the model

```bash
git clone https://github.com/amsavchenko/ntechlab-task
cd ntechlab-task/task_2
pip install -r requirements.txt
python process_folder.py /path/to/folder
```

Results of gender classification (with format *{'img_1,jpg': 'female', img_2,jpg': 'male', ...}*) will be stored into `process_results.json` file. 

---

## Structure of the repo

```
├── data                                # saved data needed for datasets loading
├── models                              # pretrained models 
├── modules
    ├── linear_classifier_cs231n.py     # for building linear classifier from scratch
    ├── load_data.py                    # for data loading and normalization
    ├── nn_models.py                    # neural net's classes (subclasses of nn.Module)
    ├── train_functions.py              # for training, cross-validation, plotting results
├── notebooks
    ├── train_cnn.ipynb                 # for train final model (CNN)
    ├── train_fc_nn.ipynb               # for train fully connected net
    ├── train_linear_classifier.ipynb   # for train ordinary linear classifier
├── config.py                           # config file with paths to used files
├── process_folder.py                   # script for scoring folder with images
```

---

## Solution's description

###  Dataset

Full dataset provided by [NtechLab](https://ntechlab.ru) consists 100k of photos equally devided between females and males. Photos have different shape and hence different quality. 

Photos were divided up into three subsets - train (15k photos), validation (3k photos) and hold (10k photos). In every subset classes are still balanced.

Data loading provided by `torchvision.datasets.ImageFolder` function.