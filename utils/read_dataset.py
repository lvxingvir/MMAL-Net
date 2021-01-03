import torch
import os
from datasets import dataset
from torchvision import transforms

def read_dataset(input_size, batch_size, root, set):
    if set == 'CUB':
        print('Loading CUB trainset')
        trainset = dataset.CUB(input_size=input_size, root=root, is_train=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=8, drop_last=False)
        print('Loading CUB testset')
        testset = dataset.CUB(input_size=input_size, root=root, is_train=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=8, drop_last=False)
    elif set == 'CAR':
        print('Loading car trainset')
        trainset = dataset.STANFORD_CAR(input_size=input_size, root=root, is_train=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=8, drop_last=False)
        print('Loading car testset')
        testset = dataset.STANFORD_CAR(input_size=input_size, root=root, is_train=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=8, drop_last=False)
    elif set == 'Aircraft':
        print('Loading Aircraft trainset')
        trainset = dataset.FGVC_aircraft(input_size=input_size, root=root, is_train=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=8, drop_last=False)
        print('Loading Aircraft testset')
        testset = dataset.FGVC_aircraft(input_size=input_size, root=root, is_train=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=8, drop_last=False)
    elif set == 'Mura':

        orisize = input_size

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((orisize, orisize)),
                # transforms.RandomResizedCrop(224),
                # transforms.CenterCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                # transforms.ColorJitter(),   # to do
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize((orisize, orisize)),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((orisize, orisize)),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        }

        print('Loading Mura')
        name = 'train'
        trainset = dataset.MURA_Dataset(data_dir=root, csv_file='MURA-v1.1/%s.csv' % name,
                                     transform=data_transforms[name])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
        print('Loading testset')

        name = 'valid'
        testset = dataset.MURA_Dataset(data_dir=root, csv_file='MURA-v1.1/%s.csv' % name,
                                        transform=data_transforms[name])
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)

    elif set == 'Mura_bp':

        orisize = input_size

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((orisize, orisize)),
                # transforms.RandomResizedCrop(224),
                # transforms.CenterCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                # transforms.ColorJitter(),   # to do
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize((orisize, orisize)),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((orisize, orisize)),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        }

        print('Loading Mura')
        name = 'train'
        trainset = dataset.MURA_Dataset(data_dir=root, csv_file='MURA-v1.1/%s_bp.csv' % name,
                                     transform=data_transforms[name])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
        print('Loading testset')

        name = 'valid'
        testset = dataset.MURA_Dataset(data_dir=root, csv_file='MURA-v1.1/%s_bp.csv' % name,
                                        transform=data_transforms[name])
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)
    else:
        print('Please choose supported dataset')
        os._exit()

    return trainloader, testloader