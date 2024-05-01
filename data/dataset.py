from torchvision import datasets
from torch.utils.data import Dataset

from .loader import pil_loader


class CustomizedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=pil_loader):
        super(CustomizedImageFolder, self).__init__(root, transform, target_transform, loader=loader)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class DatasetFromDict(Dataset):
    def __init__(self, imgs, transform=None, loader=pil_loader):
        super(DatasetFromDict, self).__init__()
        self.imgs = imgs
        self.loader = loader
        self.transform = transform
        self.targets = [img[1] for img in imgs]
        self.classes = sorted(list(set(self.targets)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = self.loader(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, label


class DatasetFromCSV(Dataset):
    def __init__(self, imgs, seqs, labels, transform=None, loader=pil_loader, crop=True):
        super(DatasetFromCSV, self).__init__()
        self.imgs = imgs  # train_set.csv, cme_days.csv, root
        self.seqs = seqs
        self.labels = labels  # cme_pics_labels.csv
        self.loader = loader
        self.transform = transform
        self.crop = crop

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        seq = self.seqs[index]
        label = self.labels.loc[img_path.split('/')[-1]].values
        img = self.loader(img_path, self.crop)

        if self.transform is not None:
            img = self.transform(img)
        return img, seq, label