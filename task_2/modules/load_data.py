import torch
from torchvision import datasets


def load_data(path_to_folder, transforms, path_to_mean_image, save_names=False):
    
    ''' Load dataset from folder, then apply transformations from torchvision.transforms module '''
    
    # load data
    dataset = datasets.ImageFolder(root=path_to_folder, transform=transforms)
    X = torch.stack([elem[0] for elem in dataset])
    y = torch.tensor([elem[1] for elem in dataset])
    
    # normalize images (mean_image was computing previously on whole 100k dataset)
    mean_image = torch.load(path_to_mean_image)
    X -= mean_image

    if save_names:
        names = [name.split('/')[-1] for name, _ in dataset.samples]
        return X, y, names
    else:
        return X, y
