"""CIFAR-adapted ResNet-18. Importable without torch; construct only in Colab."""


def build_model(settings=None):
    """Fresh non-pretrained 10-class network; never downloads ImageNet weights."""
    if settings is not None:
        if settings["model"] != "resnet18" or settings["pretrained"] or not settings["cifar_stem"]:
            raise ValueError("Expected a non-pretrained CIFAR-stem ResNet-18")
    from torch import nn
    from torchvision.models import resnet18

    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    return model
