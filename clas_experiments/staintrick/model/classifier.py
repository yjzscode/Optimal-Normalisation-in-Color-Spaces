import timm
from torch import nn
from typing import List, Optional, Tuple
from utils import init_weights, LOGITS, FEATURES


class TIMM(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: Optional[int] = 0,
        pretrained: Optional[bool] = True,
        chans: Optional[int] = 3,
    ):
        super(TIMM, self).__init__()
        self.num_classes = num_classes
        self.net = timm.create_model(model_name, pretrained=pretrained, num_classes=0,in_chans=chans)

        self.out_features = self.net.num_features

        self.head = (
            nn.Linear(self.out_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.head.apply(init_weights)

    def forward(self, x):
        features = self.net(x)
        logits = self.head(features)
        return {
            FEATURES: features,
            LOGITS: logits,
        }
