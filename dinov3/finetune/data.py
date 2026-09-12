from torchvision import datasets, transforms
from torch.utils.data import Dataset


def image_transform(size, train=False, mean=None, std=None):
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    spatial = (
        [
            transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        if train
        else [transforms.Resize(round(size * 256 / 224)), transforms.CenterCrop(size)]
    )
    return transforms.Compose(spatial + [transforms.ToTensor(), transforms.Normalize(mean, std)])


class FolderViews(Dataset):
    def __init__(self, root, transform, two_views=False):
        self.folder = datasets.ImageFolder(root)
        self.transform = transform
        self.two_views = two_views
        self.classes = self.folder.classes

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, index):
        image, label = self.folder[index]
        first = self.transform(image)
        second = self.transform(image) if self.two_views else first
        return first, second, label, index
