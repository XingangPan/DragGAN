# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Converting legacy network pickle into the new format."""

import click
import pickle
import re
import copy
import numpy as np
import torch
import dnnlib
from torch_utils import misc

#----------------------------------------------------------------------------

def load_network_pkl(f, force_fp16=False):
    data = _LegacyUnpickler(f).load()

    # Legacy TensorFlow pickle => convert.
    if isinstance(data, tuple) and len(data) == 3 and all(isinstance(net, _TFNetworkStub) for net in data):
        tf_G, tf_D, tf_Gs = data
        G = convert_tf_generator(tf_G)
        D = convert_tf_discriminator(tf_D)
        G_ema = convert_tf_generator(tf_Gs)
        data = dict(G=G, D=D, G_ema=G_ema)

    # Add missing fields.
    if 'training_set_kwargs' not in data:
        data['training_set_kwargs'] = None
    if 'augment_pipe' not in data:
        data['augment_pipe'] = None

    # Validate contents.
    assert isinstance(data['G'], torch.nn.Module)
    assert isinstance(data['D'], torch.nn.Module)
    assert isinstance(data['G_ema'], torch.nn.Module)
    assert isinstance(data['training_set_kwargs'], (dict, type(None)))
    assert isinstance(data['augment_pipe'], (torch.nn.Module, type(None)))

    # Force FP16.
    if force_fp16:
        for key in ['G', 'D', 'G_ema']:
            old = data[key]
            kwargs = copy.deepcopy(old.init_kwargs)
            fp16_kwargs = kwargs.get('synthesis_kwargs', kwargs)
            fp16_kwargs.num_fp16_res = 4
            fp16_kwargs.conv_clamp = 256
            if kwargs != old.init_kwargs:
                new = type(old)(**kwargs).eval().requires_grad_(False)
                misc.copy_params_and_buffers(old, new, require_all=True)
                data[key] = new
    return data

#----------------------------------------------------------------------------

class _TFNetworkStub(dnnlib.EasyDict):
    pass

class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'dnnlib.tflib.network' and name == 'Network':
            return _TFNetworkStub
        return super().find_class(module, name)

#----------------------------------------------------------------------------

def _collect_tf_params(tf_net):
    # pylint: disable=protected-access
    tf_params = dict()
    def recurse(prefix, tf_net):
        for name, value in tf_net.variables:
            tf_params[prefix + name] = value
        for name, comp in tf_net.components.items():
            recurse(prefix + name + '/', comp)
    recurse('', tf_net)
    return tf_params

#----------------------------------------------------------------------------

def _populate_module_params(module, *patterns):
    for name, tensor in misc.named_params_and_buffers(module):
        found = False
        value = None
        for pattern, value_fn in zip(patterns[0::2], patterns[1::2]):
            match = re.fullmatch(pattern, name)
            if match:
                found = True
                if value_fn is not None:
                    value = value_fn(*match.groups())
                break
        try:
            assert found
            if value is not None:
                tensor.copy_(torch.from_numpy(np.array(value)))
        except:
            print(name, list(tensor.shape))
            raise

#----------------------------------------------------------------------------

def convert_tf_generator(tf_G):
    if tf_G.version < 4:
        raise ValueError('TensorFlow pickle version too low')

    # Collect kwargs.
    tf_kwargs = tf_G.static_kwargs
    known_kwargs = set()
    def kwarg(tf_name, default=None, none=None):
        known_kwargs.add(tf_name)
        val = tf_kwargs.get(tf_name, default)
        return val if val is not None else none

    # Convert kwargs.
    from training import networks_stylegan2
    network_class = networks_stylegan2.Generator
    kwargs = dnnlib.EasyDict(
        z_dim               = kwarg('latent_size',          512),
        c_dim               = kwarg('label_size',           0),
        w_dim               = kwarg('dlatent_size',         512),
        img_resolution      = kwarg('resolution',           1024),
        img_channels        = kwarg('num_channels',         3),
        channel_base        = kwarg('fmap_base',            16384) * 2,
        channel_max         = kwarg('fmap_max',             512),
        num_fp16_res        = kwarg('num_fp16_res',         0),
        conv_clamp          = kwarg('conv_clamp',           None),
        architecture        = kwarg('architecture',         'skip'),
        resample_filter     = kwarg('resample_kernel',      [1,3,3,1]),
        use_noise           = kwarg('use_noise',            True),
        activation          = kwarg('nonlinearity',         'lrelu'),
        mapping_kwargs      = dnnlib.EasyDict(
            num_layers      = kwarg('mapping_layers',       8),
            embed_features  = kwarg('label_fmaps',          None),
            layer_features  = kwarg('mapping_fmaps',        None),
            activation      = kwarg('mapping_nonlinearity', 'lrelu'),
            lr_multiplier   = kwarg('mapping_lrmul',        0.01),
            w_avg_beta      = kwarg('w_avg_beta',           0.995,  none=1),
        ),
    )

    # Check for unknown kwargs.
    kwarg('truncation_psi')
    kwarg('truncation_cutoff')
    kwarg('style_mixing_prob')
    kwarg('structure')
    kwarg('conditioning')
    kwarg('fused_modconv')
    unknown_kwargs = list(set(tf_kwargs.keys()) - known_kwargs)
    if len(unknown_kwargs) > 0:
        raise ValueError('Unknown TensorFlow kwarg', unknown_kwargs[0])

    # Collect params.
    tf_params = _collect_tf_params(tf_G)
    for name, value in list(tf_params.items()):
        match = re.fullmatch(r'ToRGB_lod(\d+)/(.*)', name)
        if match:
            r = kwargs.img_resolution // (2 ** int(match.group(1)))
            tf_params[f'{r}x{r}/ToRGB/{match.group(2)}'] = value
            kwargs.synthesis.kwargs.architecture = 'orig'
    #for name, value in tf_params.items(): print(f'{name:<50s}{list(value.shape)}')

    # Convert params.
    G = network_class(**kwargs).eval().requires_grad_(False)
    # pylint: disable=unnecessary-lambda
    # pylint: disable=f-string-without-interpolation
    _populate_module_params(G,
        r'mapping\.w_avg',                                  lambda:     tf_params[f'dlatent_avg'],
        r'mapping\.embed\.weight',                          lambda:     tf_params[f'mapping/LabelEmbed/weight'].transpose(),
        r'mapping\.embed\.bias',                            lambda:     tf_params[f'mapping/LabelEmbed/bias'],
        r'mapping\.fc(\d+)\.weight',                        lambda i:   tf_params[f'mapping/Dense{i}/weight'].transpose(),
        r'mapping\.fc(\d+)\.bias',                          lambda i:   tf_params[f'mapping/Dense{i}/bias'],
        r'synthesis\.b4\.const',                            lambda:     tf_params[f'synthesis/4x4/Const/const'][0],
        r'synthesis\.b4\.conv1\.weight',                    lambda:     tf_params[f'synthesis/4x4/Conv/weight'].transpose(3, 2, 0, 1),
        r'synthesis\.b4\.conv1\.bias',                      lambda:     tf_params[f'synthesis/4x4/Conv/bias'],
        r'synthesis\.b4\.conv1\.noise_const',               lambda:     tf_params[f'synthesis/noise0'][0, 0],
        r'synthesis\.b4\.conv1\.noise_strength',            lambda:     tf_params[f'synthesis/4x4/Conv/noise_strength'],
        r'synthesis\.b4\.conv1\.affine\.weight',            lambda:     tf_params[f'synthesis/4x4/Conv/mod_weight'].transpose(),
        r'synthesis\.b4\.conv1\.affine\.bias',              lambda:     tf_params[f'synthesis/4x4/Conv/mod_bias'] + 1,
        r'synthesis\.b(\d+)\.conv0\.weight',                lambda r:   tf_params[f'synthesis/{r}x{r}/Conv0_up/weight'][::-1, ::-1].transpose(3, 2, 0, 1),
        r'synthesis\.b(\d+)\.conv0\.bias',                  lambda r:   tf_params[f'synthesis/{r}x{r}/Conv0_up/bias'],
        r'synthesis\.b(\d+)\.conv0\.noise_const',           lambda r:   tf_params[f'synthesis/noise{int(np.log2(int(r)))*2-5}'][0, 0],
        r'synthesis\.b(\d+)\.conv0\.noise_strength',        lambda r:   tf_params[f'synthesis/{r}x{r}/Conv0_up/noise_strength'],
        r'synthesis\.b(\d+)\.conv0\.affine\.weight',        lambda r:   tf_params[f'synthesis/{r}x{r}/Conv0_up/mod_weight'].transpose(),
        r'synthesis\.b(\d+)\.conv0\.affine\.bias',          lambda r:   tf_params[f'synthesis/{r}x{r}/Conv0_up/mod_bias'] + 1,
        r'synthesis\.b(\d+)\.conv1\.weight',                lambda r:   tf_params[f'synthesis/{r}x{r}/Conv1/weight'].transpose(3, 2, 0, 1),
        r'synthesis\.b(\d+)\.conv1\.bias',                  lambda r:   tf_params[f'synthesis/{r}x{r}/Conv1/bias'],
        r'synthesis\.b(\d+)\.conv1\.noise_const',           lambda r:   tf_params[f'synthesis/noise{int(np.log2(int(r)))*2-4}'][0, 0],
        r'synthesis\.b(\d+)\.conv1\.noise_strength',        lambda r:   tf_params[f'synthesis/{r}x{r}/Conv1/noise_strength'],
        r'synthesis\.b(\d+)\.conv1\.affine\.weight',        lambda r:   tf_params[f'synthesis/{r}x{r}/Conv1/mod_weight'].transpose(),
        r'synthesis\.b(\d+)\.conv1\.affine\.bias',          lambda r:   tf_params[f'synthesis/{r}x{r}/Conv1/mod_bias'] + 1,
        r'synthesis\.b(\d+)\.torgb\.weight',                lambda r:   tf_params[f'synthesis/{r}x{r}/ToRGB/weight'].transpose(3, 2, 0, 1),
        r'synthesis\.b(\d+)\.torgb\.bias',                  lambda r:   tf_params[f'synthesis/{r}x{r}/ToRGB/bias'],
        r'synthesis\.b(\d+)\.torgb\.affine\.weight',        lambda r:   tf_params[f'synthesis/{r}x{r}/ToRGB/mod_weight'].transpose(),
        r'synthesis\.b(\d+)\.torgb\.affine\.bias',          lambda r:   tf_params[f'synthesis/{r}x{r}/ToRGB/mod_bias'] + 1,
        r'synthesis\.b(\d+)\.skip\.weight',                 lambda r:   tf_params[f'synthesis/{r}x{r}/Skip/weight'][::-1, ::-1].transpose(3, 2, 0, 1),
        r'.*\.resample_filter',                             None,
        r'.*\.act_filter',                                  None,
    )
    return G

#----------------------------------------------------------------------------

def convert_tf_discriminator(tf_D):
    if tf_D.version < 4:
        raise ValueError('TensorFlow pickle version too low')

    # Collect kwargs.
    tf_kwargs = tf_D.static_kwargs
    known_kwargs = set()
    def kwarg(tf_name, default=None):
        known_kwargs.add(tf_name)
        return tf_kwargs.get(tf_name, default)

    # Convert kwargs.
    kwargs = dnnlib.EasyDict(
        c_dim                   = kwarg('label_size',           0),
        img_resolution          = kwarg('resolution',           1024),
        img_channels            = kwarg('num_channels',         3),
        architecture            = kwarg('architecture',         'resnet'),
        channel_base            = kwarg('fmap_base',            16384) * 2,
        channel_max             = kwarg('fmap_max',             512),
        num_fp16_res            = kwarg('num_fp16_res',         0),
        conv_clamp              = kwarg('conv_clamp',           None),
        cmap_dim                = kwarg('mapping_fmaps',        None),
        block_kwargs = dnnlib.EasyDict(
            activation          = kwarg('nonlinearity',         'lrelu'),
            resample_filter     = kwarg('resample_kernel',      [1,3,3,1]),
            freeze_layers       = kwarg('freeze_layers',        0),
        ),
        mapping_kwargs = dnnlib.EasyDict(
            num_layers          = kwarg('mapping_layers',       0),
            embed_features      = kwarg('mapping_fmaps',        None),
            layer_features      = kwarg('mapping_fmaps',        None),
            activation          = kwarg('nonlinearity',         'lrelu'),
            lr_multiplier       = kwarg('mapping_lrmul',        0.1),
        ),
        epilogue_kwargs = dnnlib.EasyDict(
            mbstd_group_size    = kwarg('mbstd_group_size',     None),
            mbstd_num_channels  = kwarg('mbstd_num_features',   1),
            activation          = kwarg('nonlinearity',         'lrelu'),
        ),
    )

    # Check for unknown kwargs.
    kwarg('structure')
    kwarg('conditioning')
    unknown_kwargs = list(set(tf_kwargs.keys()) - known_kwargs)
    if len(unknown_kwargs) > 0:
        raise ValueError('Unknown TensorFlow kwarg', unknown_kwargs[0])

    # Collect params.
    tf_params = _collect_tf_params(tf_D)
    for name, value in list(tf_params.items()):
        match = re.fullmatch(r'FromRGB_lod(\d+)/(.*)', name)
        if match:
            r = kwargs.img_resolution // (2 ** int(match.group(1)))
            tf_params[f'{r}x{r}/FromRGB/{match.group(2)}'] = value
            kwargs.architecture = 'orig'
    #for name, value in tf_params.items(): print(f'{name:<50s}{list(value.shape)}')

    # Convert params.
    from training import networks_stylegan2
    D = networks_stylegan2.Discriminator(**kwargs).eval().requires_grad_(False)
    # pylint: disable=unnecessary-lambda
    # pylint: disable=f-string-without-interpolation
    _populate_module_params(D,
        r'b(\d+)\.fromrgb\.weight',     lambda r:       tf_params[f'{r}x{r}/FromRGB/weight'].transpose(3, 2, 0, 1),
        r'b(\d+)\.fromrgb\.bias',       lambda r:       tf_params[f'{r}x{r}/FromRGB/bias'],
        r'b(\d+)\.conv(\d+)\.weight',   lambda r, i:    tf_params[f'{r}x{r}/Conv{i}{["","_down"][int(i)]}/weight'].transpose(3, 2, 0, 1),
        r'b(\d+)\.conv(\d+)\.bias',     lambda r, i:    tf_params[f'{r}x{r}/Conv{i}{["","_down"][int(i)]}/bias'],
        r'b(\d+)\.skip\.weight',        lambda r:       tf_params[f'{r}x{r}/Skip/weight'].transpose(3, 2, 0, 1),
        r'mapping\.embed\.weight',      lambda:         tf_params[f'LabelEmbed/weight'].transpose(),
        r'mapping\.embed\.bias',        lambda:         tf_params[f'LabelEmbed/bias'],
        r'mapping\.fc(\d+)\.weight',    lambda i:       tf_params[f'Mapping{i}/weight'].transpose(),
        r'mapping\.fc(\d+)\.bias',      lambda i:       tf_params[f'Mapping{i}/bias'],
        r'b4\.conv\.weight',            lambda:         tf_params[f'4x4/Conv/weight'].transpose(3, 2, 0, 1),
        r'b4\.conv\.bias',              lambda:         tf_params[f'4x4/Conv/bias'],
        r'b4\.fc\.weight',              lambda:         tf_params[f'4x4/Dense0/weight'].transpose(),
        r'b4\.fc\.bias',                lambda:         tf_params[f'4x4/Dense0/bias'],
        r'b4\.out\.weight',             lambda:         tf_params[f'Output/weight'].transpose(),
        r'b4\.out\.bias',               lambda:         tf_params[f'Output/bias'],
        r'.*\.resample_filter',         None,
    )
    return D

#----------------------------------------------------------------------------

@click.command()
@click.option('--source', help='Input pickle', required=True, metavar='PATH')
@click.option('--dest', help='Output pickle', required=True, metavar='PATH')
@click.option('--force-fp16', help='Force the networks to use FP16', type=bool, default=False, metavar='BOOL', show_default=True)
def convert_network_pickle(source, dest, force_fp16):
    """Convert legacy network pickle into the native PyTorch format.

    The tool is able to load the main network configurations exported using the TensorFlow version of StyleGAN2 or StyleGAN2-ADA.
    It does not support e.g. StyleGAN2-ADA comparison methods, StyleGAN2 configs A-D, or StyleGAN1 networks.

    Example:

    \b
    python legacy.py \\
        --source=https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-cat-config-f.pkl \\
        --dest=stylegan2-cat-config-f.pkl
    """
    print(f'Loading "{source}"...')
    with dnnlib.util.open_url(source) as f:
        data = load_network_pkl(f, force_fp16=force_fp16)
    print(f'Saving "{dest}"...')
    with open(dest, 'wb') as f:
        pickle.dump(data, f)
    print('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    convert_network_pickle() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
