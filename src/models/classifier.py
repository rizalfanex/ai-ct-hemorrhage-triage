from typing import Optional

import timm
import torch
import torch.nn as nn


class RSNAHemorrhageClassifier(nn.Module):
    """
    Multi-label classifier for RSNA Intracranial Hemorrhage Detection.

    Input:
        image tensor: [B, 3, H, W]

    Output:
        logits: [B, 6]
        labels:
            0 any
            1 epidural
            2 intraparenchymal
            3 intraventricular
            4 subarachnoid
            5 subdural
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 6,
        pretrained: bool = False,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()

        kwargs = {
            "model_name": model_name,
            "pretrained": pretrained,
            "num_classes": num_classes,
            "in_chans": 3,
        }

        if dropout is not None:
            kwargs["drop_rate"] = dropout

        self.model = timm.create_model(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_model(
    model_name: str = "efficientnet_b0",
    num_classes: int = 6,
    pretrained: bool = False,
) -> RSNAHemorrhageClassifier:
    return RSNAHemorrhageClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )
