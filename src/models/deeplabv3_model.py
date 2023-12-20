import torchvision.models as models
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
import torch


def get_deeplabv3_pretrained(num_classes):
    """
    Load a pre-trained DeepLabV3 model with a specified number of output classes.

    Args:
    - num_classes (int): Number of classes for the final classification layer.

    Returns:
    - model: A DeepLabV3 model with the final layer adjusted to the number of classes.
    """
    # Load pre-trained DeepLabV3 model
    weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
    model = models.segmentation.deeplabv3_resnet101(weights=weights)

    # Modify the classifier to have the correct number of classes
    model.classifier[4] = torch.nn.Conv2d(
        256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    return model


if __name__ == "__main__":
    # Example usage
    num_classes = 21  # Example: number of classes in your dataset
    model = get_deeplabv3_pretrained(num_classes)
    print(model)
