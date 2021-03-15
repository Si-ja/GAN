import torch
import torchvision
from torchvision import transforms

def prepare_tranformer():
    """
    A function that generates a data transoformer that augments data
    while the networks train.

    Returns
    -------
    transform : TYPE
        transform (obj) which transforms the data to the format 
        most suitable for the Neural Networks to use

    """
    transform = transforms.Compose([
                            transforms.RandomRotation(degrees=(0, 60)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5],
                            std=[0.5])
                                  ])
    
    return transform

def load_data(transform, data_path):    
    """
    A function that loads data from the MNIST dataset.
    
    Parameters
    ----------
    transform : obj
        An object that can perofrm transformation on the loaded MNIST data.
    data_path : str
        An indication where the MNIST data is stored and where it can be 
        retrieved from on the users computer.

    Returns
    -------
    mnist data - can be pushed further into a data loader.
    """
    mnist = torchvision.datasets.MNIST(root=data_path,
                                       train=True,
                                       transform=transform,
                                       download=True)
    
    return mnist

def prepare_loader(dataset, batch_size, shuffle=True):
    """
    A function that prepares a data loader that can be passed to the
    neural networks and allow for them to train in itterations.
    
    Parameters
    ----------
    dataset : obj
        A reference to the dataset that is used in the training process.
    batch_size : int
        A size of the batch size with which the training of the models happen.
    shuffle : bool
        A reference to whether the data should be shuffled and given to the
        networks to train with or not. Defaults to: True.
        
    Returns
    -------
    data_loader - and object with which data can be iteratively read by
                  the neural networks.
    """
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    
    return data_loader

def generate_loader(data_path, batch_size=64, shuffle=True):
    """
    A function that prepares the data loaders with all the necessary
    parameters right from the start, including the moments where the
    data needs to be also prepared. Exists for convenience.
    
    Parameters
    ----------
    batch_size : int
        A size of the batch size with which the training of the models happen.
    shuffle : bool
        A reference to whether the data should be shuffled and given to the
        networks to train with or not. Defaults to: True.
        
    Returns
    -------
    data_loader - and object with which data can be iteratively read by
                  the neural networks.
    """
    transform = prepare_tranformer()
    mnist = load_data(transform=transform, data_path=data_path)
    data_loader = prepare_loader(dataset=mnist, batch_size=batch_size, shuffle=shuffle)
    
    return data_loader