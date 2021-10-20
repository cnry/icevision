__all__ = ["model"]
from icevision.utils import *
from icevision.imports import *
from icevision.models.torchvision.utils import *
from icevision.models.torchvision.backbones.backbone_config import (
    TorchvisionBackboneConfig,
)
from torchvision.models.detection.ssdlite import (
    ssdlite320_mobilenet_v3_large,
    SSD,
    SSDLiteHead,
)


def mobilenet_param_groups(model: nn.Module) -> List[List[nn.Parameter]]:
    model.parameters()
    param_groups = [[*model.parameters()]]
    check_all_model_params_in_groups2(model, param_groups)

    return param_groups

def model(
    num_classes: int,
    backbone: Optional[TorchvisionBackboneConfig] = None,
    remove_internal_transforms: bool = True,
    pretrained: bool = True,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    **ssdlite_kwargs
) -> nn.Module:
    model = ssdlite320_mobilenet_v3_large(
                pretrained_backbone=pretrained, 
                num_classes=num_classes,
                **ssdlite_kwargs
            )

    features = model.backbone
    features.out_channels = 1280
    features.param_groups = MethodType(mobilenet_param_groups, features)
    
    patch_ssdlite_param_groups(model)

    if remove_internal_transforms:
        remove_internal_model_transforms(model)

    return model
