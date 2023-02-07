import torch
from torchvision.datasets import ImageFolder
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")


class HistoDataset(ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        transform_list=None,
    ):

        super(HistoDataset, self).__init__(root=root, transform=transform)
        self.transform_list = transform_list

    def __getitem__(self, index):

        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform_list is not None:
            img_transformed = []
            for transform in self.transform_list:
                img_transformed.append(transform(img.copy()))
            img = torch.stack(img_transformed)
        elif self.transform is not None:
            img = self.transform(img)
        return img, target
