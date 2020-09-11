import glob
import json
import os
import shutil
import sys

import torch
from torchvision import transforms

from config import path_to_mean_image, path_to_model
from modules.load_data import load_data
from modules.nn_models import ConvolutionalNN

path_to_root = './'
path_to_mean_image = path_to_root + path_to_mean_image
path_to_model = path_to_root + path_to_model

spatial_shape = 64
test_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((spatial_shape, spatial_shape)),
    transforms.ColorJitter(),
    transforms.ToTensor(),
])

net = ConvolutionalNN()
net.load_state_dict(torch.load(path_to_model))


def predict(model, X):
    ''' Predict labels for X '''
    out = model(X)
    out = (out.flatten() > 0.5).float()
    return out


def to_class_name(x):
    return 'female' if x == 0.0 else 'male'


if __name__ == "__main__":
    # read path to folder with images
    try:
        data_dir = sys.argv[1]
    except IndexError:
        print('Pass a path to a folder with images as a command line argument \n'
              'Example: python process_folder.py ./test_data \n')
        quit()
    
    # find subdirectories of data_dir 
    subdirs = [os.path.join(data_dir, o) for o in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, o))]
    
    # if data_dir hasn't subdirectories, then we must create one
    # and move all images into it (neccessary for torchvision.datasets.ImageFolder)
    if len(subdirs) == 0:
        subdir = os.path.join(data_dir, '1')
        os.makedirs(subdir)
        for pic in glob.iglob(os.path.join(data_dir, "*.*")):
            shutil.move(pic, subdir)

    # now when data_dir has a subdirectory we can load data
    X, _, names = load_data(data_dir, test_transforms,
                            path_to_mean_image, save_names=True)
    
    # get pridicted labels for X
    y_pred = predict(net, X)

    # combine names and y_pred into a dictionary
    answers = {name: to_class_name(out.item()) for name, out in zip(names, list(y_pred))}

    with open(path_to_root + 'process_results.json', 'w') as f:
        json.dump(answers, f)

    print(f'Result of processing {len(names)} images stored into process_results.json\n')
