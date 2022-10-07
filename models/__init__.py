import torch
import torchvision
from .add_gcn import ADD_GCN
from .resnet101_gap import ResNet101_GAP
from .resnet101_gmp import ResNet101_GMP
model_dict = {'ResNet101_GAP': ResNet101_GAP,
              'ResNet101_GMP': ResNet101_GMP,
              'ADD_GCN': ADD_GCN,
                }

def get_model(num_classes, args):
    from torchvision.models import ResNet101_Weights
    if args.imagenet in ('v1', 'V1'):
        # res101 = torchvision.models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        res101 = torchvision.models.resnet101(pretrained=True)
    elif args.imagenet in ('v2', 'V2'):
        res101 = torchvision.models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    else:
        raise NotImplementedError('Only IMAGENET1K_V1 or IMAGENET1K_V2 can be chosen!')
    model = model_dict[args.model_name](res101, num_classes)
    return model