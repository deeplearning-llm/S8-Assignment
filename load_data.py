from torchvision import datasets, transforms
import torch

def cifar10_train_test_data(batch_size_=128):

    # Train Phase transformations
    train_transforms = transforms.Compose([
                                          #  transforms.Resize((28, 28)),
                                        #   transforms.ColorJitter(brightness=0.10, contrast=0.5, saturation=0.10, hue=0.1),
                                       #   transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                                           transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
                                           # Note the difference between (0.1307) and (0.1307,)
                                           ])

    # Test Phase transformations
    test_transforms = transforms.Compose([
                                          #  transforms.Resize((28, 28)),
                                          #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                           transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                           ])
                                           
    train = datasets.CIFAR10('./data', train=True, download=True, transform=train_transforms)
    test = datasets.CIFAR10('./data', train=False, download=True, transform=test_transforms)
    
    SEED = 1
    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=batch_size_, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(test, **dataloader_args)
    
    return train_loader,test_loader

