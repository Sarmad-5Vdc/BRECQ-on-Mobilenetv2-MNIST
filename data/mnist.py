import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import os
import nltk



def build_mnist_data(data_path: str = '', input_size: int = (28,28), batch_size: int = 32, workers: int = 4,
                        dist_sample: bool = False):
    data_path = './'
    input_size = 28

    normalize = transforms.Normalize(mean=[0.1307],
                                     std=[0.3081])

    train_dataset = datasets.MNIST(data_path,train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    
    val_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(), normalize]))
   
    torchvision.set_image_backend('accimage')


    if dist_sample:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=val_sampler)
    return train_loader, val_loader
