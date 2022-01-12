import pytest
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

#########################################---TESTING---##################################################
# the next portion of the script will be testing some key point about the data, such as dimensionlity, #
# size, label representation, etc. The data loaded will not be local but will be loaded from pytorch   #
# module, this is done since the self made dataloader is to time consuming and is not the point of the #
# exercise in the first place.                                                                         #
########################################################################################################
def get_train():
    mnist_transform = transforms.Compose([transforms.ToTensor()])
    dataset_path = 'datasets'
    train_dataset = MNIST(dataset_path,
                    transform=mnist_transform,
                    train=True,
                    download=True)
    return train_dataset

def get_test():
    mnist_transform = transforms.Compose([transforms.ToTensor()])
    dataset_path = 'datasets'
    test_dataset = MNIST(dataset_path,
                        transform=mnist_transform,
                        train=False,
                        download=True)
    return test_dataset

def test_train_len():
    train_dataset=get_train()
    N_train = 60000
    assert len(train_dataset) == N_train


def test_len_answ():
    test_dataset=get_test()
    N_test = 10000
    assert len(test_dataset) == N_test


def test_dimnesion():
    dimension = torch.Size([28, 28])
    test_dataset=get_test()
    for i in range(len(test_dataset)):
        assert test_dataset.data[i].shape == dimension,f"datapoint {i} does not corrospond with the input dimesion {dimension}"


def test_label_rep():
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    train_dataset=get_train()
    ## input all the expected labels for the data, will throw error if a label in the labels array is not present in the dataset
    ## this array can be made into a tuble if string labels are required. 
    for index, value in enumerate(labels):
        assert value in train_dataset.targets, f" {value} is not present in the dataset" 


@pytest.mark.parametrize("dim1,dim2,expected", [(28, 28, 784), (10, 10, 100), (32, 32, 1024)])
def test_dim_flatten(dim1, dim2, expected):
    temp_tens = torch.rand(dim1, dim2)
    assert torch.flatten(temp_tens).size(dim=0) == expected, f"input dimension: [{dim1}, {dim2}] did not meet expecteed output size after flattening of {expected}."
