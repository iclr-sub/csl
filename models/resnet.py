import torch.nn as nn
import torchvision.models as models

class ResNet32(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(ResNet32, self).__init__()
        self.model = models.resnet34(pretrained=pretrained)
        self._update_num_classes(num_classes)

    def _update_num_classes(self, num_classes):
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        self._update_num_classes(num_classes)

    def _update_num_classes(self, num_classes):
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class ResNet101(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False):
        super(ResNet101, self).__init__()
        self.model = models.resnet101(pretrained=pretrained)
        self._update_num_classes(num_classes)

    def _update_num_classes(self, num_classes):
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
