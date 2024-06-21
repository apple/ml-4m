# --------------------------------------------------------
# Based on the timm code base
# https://github.com/huggingface/pytorch-image-models
# --------------------------------------------------------
from .registry import is_model_in_modules, model_entrypoint


def split_model_name(model_name):
    model_split = model_name.split(':', 1)
    if len(model_split) == 1:
        return '', model_split[0]
    else:
        source_name, model_name = model_split
        assert source_name in ('timm', 'hf_hub')
        return source_name, model_name


def safe_model_name(model_name, remove_source=True):
    def make_safe(name):
        return ''.join(c if c.isalnum() else '_' for c in name).rstrip('_')

    if remove_source:
        model_name = split_model_name(model_name)[-1]
    return make_safe(model_name)


def create_model(
        model_name,
        pretrained=False,
        checkpoint_path='',
        scriptable=None,
        exportable=None,
        no_jit=None,
        **kwargs):
    """Create a model

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        checkpoint_path (str): path of checkpoint to load after model is initialized
        scriptable (bool): set layer config so that model is jit scriptable (not working for all models yet)
        exportable (bool): set layer config so that model is traceable / ONNX exportable (not fully impl/obeyed yet)
        no_jit (bool): set layer config so that model doesn't utilize jit scripted layers (so far activations only)

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific
    """
    source_name, model_name = split_model_name(model_name)

    # Only EfficientNet and MobileNetV3 models have support for batchnorm params or drop_connect_rate passed as args
    is_efficientnet = is_model_in_modules(model_name, ['efficientnet', 'mobilenetv3'])
    if not is_efficientnet:
        kwargs.pop('bn_tf', None)
        kwargs.pop('bn_momentum', None)
        kwargs.pop('bn_eps', None)

    # handle backwards compat with drop_connect -> drop_path change
    drop_connect_rate = kwargs.pop('drop_connect_rate', None)
    if drop_connect_rate is not None and kwargs.get('drop_path_rate', None) is None:
        print("WARNING: 'drop_connect' as an argument is deprecated, please use 'drop_path'."
              " Setting drop_path to %f." % drop_connect_rate)
        kwargs['drop_path_rate'] = drop_connect_rate

    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items()}

    create_fn = model_entrypoint(model_name)
    model = create_fn(**kwargs)

    return model
