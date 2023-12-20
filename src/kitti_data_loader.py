import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


class KITTIDataset(Dataset):
    """
    A PyTorch dataset class for loading KITTI dataset for pixel-level semantic segmentation.
    """

    def __init__(self, image_transform=None, mask_transform=None):
        """
        Initializes the KITTIDataset class.

        Args:
            image_transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Default is None.
            mask_transform (callable, optional): A function/transform that takes in a mask and returns a transformed version. Default is None.
        """
        pass
        current_dir = os.path.dirname(__file__)
        data_dir = os.path.join(
            current_dir, '../data/data_semantics/training')

        self.data_dir = data_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.images = os.listdir(os.path.join(data_dir, 'image_2'))

    def __len__(self):
        """
        Returns the length of the data loader.

        Returns:
            int: The number of images in the data loader.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieves the image and mask at the given index.

        Args:
            idx (int): Index of the image and mask to retrieve.

        Returns:
            tuple: A tuple containing the image and mask.
        """

        img_name = self.images[idx]
        img_path = os.path.join(self.data_dir, 'image_2', img_name)
        mask_path = os.path.join(self.data_dir, 'semantic', img_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Convert the PIL Image mask to a long tensor. This preserves the IDs
        # as ints by preventing them from being normalized from 0 to 1
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask


if __name__ == '__main__':
    resize = transforms.Resize(256)
    crop = transforms.CenterCrop((224, 224))
    image_transform = transforms.Compose([
        resize,
        crop,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        resize,
        crop,
    ])

    dataset = KITTIDataset(image_transform=image_transform,
                           mask_transform=mask_transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(dataloader))
    images, masks = batch

    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(10, 20))
    for i in range(4):
        ax[i, 0].imshow(images[i].permute(1, 2, 0))
        ax[i, 1].imshow(masks[i].squeeze())
        ax[i, 0].set_title("Image")
        ax[i, 1].set_title("Mask")
        ax[i, 0].axis("off")
        ax[i, 1].axis("off")

    plt.show()
