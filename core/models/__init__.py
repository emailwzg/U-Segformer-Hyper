import imp
import torchvision.models as models
from core.models.patch_deconvnet import *
from core.models.section_deconvnet import *
from core.models import segformers


def get_model(name, pretrained, n_classes, **kwargs):
    model = _get_model_instance(name)

    if name in ['section_deconvnet','patch_deconvnet']:
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=pretrained)
        model.init_vgg16_params(vgg16)
    elif 'unet' in name:
        model = model(**kwargs)
    elif 'seg' in name:
        model = model(**kwargs)
    else:
        model = model(n_classes=n_classes, **kwargs)

    return model

def _get_model_instance(name):
    try:
        return {
            'section_deconvnet': section_deconvnet,
            'patch_deconvnet':patch_deconvnet,
            'section_deconvnet_skip': section_deconvnet_skip,
            'patch_deconvnet_skip':patch_deconvnet_skip,
            'segformer': segformers.oriSegformer,
            'usegformer': segformers.USegformerLi,
            'usegformerhyper': segformers.USegformerHyper,

        }[name]
    except:
        print(f'Model {name} not available')
