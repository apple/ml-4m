# --------------------------------------------------------
# Based on the ViTPose and 4DHumans code bases
# https://github.com/ViTAE-Transformer/ViTPose/
# https://github.com/shubham-goel/4D-Humans
# --------------------------------------------------------

from .vit import vit

def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'vit':
        return vit(cfg)
    else:
        raise NotImplementedError('Backbone type is not implemented')
