# Task 2 - Gender Classification

A model trained to classify a person's gender by a photo.

## How to use the model

To try gender classification collect photos in some directory `/path/to/folder` and then use following commands:

```bash
git clone https://github.com/amsavchenko/ntechlab-task
cd ntechlab-task/task_2
pip install -r requirements.txt
python process_folder.py /path/to/folder
```

Results of gender classification (with format *{'img_1,jpg': 'female', img_2,jpg': 'male', ...}*) will be stored into `process_results.json` file. 

To run again training process of the model launch `notebooks/train_cnn.ipynb`.

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
    ├── train_cnn.ipynb                 # for train the final model (CNN)
    ├── train_fc_nn.ipynb               # for train the fully connected net
    ├── train_linear_classifier.ipynb   # for train the logistic regression
├── config.py                           # config file with paths to used files
├── process_folder.py                   # script for scoring folder with images
```

---

## Solution's description

###  Dataset

Full dataset provided by [NtechLab](https://ntechlab.ru) consists 100k of photos equally divided between females and males. Photos have different shape and hence quality. 

Photos were divided up into three subsets - train (15k photos), validation (3k photos) and hold (10k photos). In every subset classes are still balanced.

Data loading provided by `torchvision.datasets.ImageFolder` function (more in [load_data.py](https://github.com/amsavchenko/ntechlab-task/blob/master/task_2/modules/load_data.py)).

All photos were process with following set of transformations:

- Grayscale
- Resize
- ColorJitter
- RandomHorizontalCrop

Transformations described in detail in every notebook in the `notebook/` directory.

### Models and training 

I applied 3 types of models for this problem:

1. Logistic Regression – [notebook](https://github.com/amsavchenko/ntechlab-task/blob/master/task_2/notebooks/train_linear_classifier.ipynb)
2. Fully-Connected Neural Network – [notebook](https://github.com/amsavchenko/ntechlab-task/blob/master/task_2/notebooks/train_fc_nn.ipynb)
3. Convolutional Neural Network – [notebook](https://github.com/amsavchenko/ntechlab-task/blob/master/task_2/notebooks/train_cnn.ipynb) (**final model**) 

For hyperparameters searching I use `cross_validation_score` from [train_functions.py](https://github.com/amsavchenko/ntechlab-task/blob/master/task_2/modules/train_functions.py) with 3 or 5 folds. For all models the best optimizer was `Adam` with `lr=1e-3` and `betas=(0.9, 0.999)`. 

Results for the final CNN model:

- `0.9275` on cross-validation
- `0.9633` on validation data when model trained on the train dataset for 10 epochs
- `0.9531` on hold (unseen) data

More detailed description of training process, net's architectures and results provided in each notebook.





