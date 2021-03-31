from torch.utils.data import DataLoader, Dataset
import glob
import os
import torch
from PIL import Image


class Edge2MogaokuDataLoader(object):

    def __init__(self, data_root, batch_size, train_num_workers, transforms=None, test_num_workers=0):

        train_root = os.path.join(data_root, "train")
        test_root = os.path.join(data_root, "val")

        train_image_list = glob.glob(os.path.join(train_root, "*"))
        test_image_list = glob.glob(os.path.join(test_root, "*"))

        self.train_dataset = Edge2MogaokuDataSet(train_image_list, transforms=transforms)
        self.test_dataset = Edge2MogaokuDataSet(test_image_list, transforms=transforms)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size, shuffle=True,
                                           drop_last=True, num_workers=train_num_workers,
                                           collate_fn=self.train_dataset.collate_fn)
        self.test_dataloader = DataLoader(self.test_dataset, 1, shuffle=False,
                                          num_workers=test_num_workers,
                                          collate_fn=self.test_dataset.collate_fn)


class Edge2MogaokuDataSet(Dataset):

    def __init__(self, data, transforms=None):
        super(Edge2MogaokuDataSet, self).__init__()

        self.train_list = data
        self.transforms = transforms

    def __getitem__(self, item):

        image = Image.open(self.train_list[item])

        image_edge = image.crop([0, 0, image.size[0] / 2, image.size[1]])
        image_color = image.crop([image.size[0]/2, 0, image.size[0], image.size[1]])

        if self.transforms is not None:
            image_edge = self.transforms(image_edge)
            image_color = self.transforms(image_color)

        return {"edge_image": image_edge, "color_image": image_color}

    def __len__(self):
        return len(self.train_list)

    def collate_fn(self, batch):

        edge_images = torch.cat([torch.unsqueeze(s["edge_image"], 0) for s in batch], dim=0).type(torch.float32)
        color_images = torch.cat([torch.unsqueeze(s["color_image"], 0) for s in batch], dim=0).type(torch.float32)
        return {"edge_images": edge_images, "color_images": color_images}
