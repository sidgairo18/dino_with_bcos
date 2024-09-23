"""
From lucidrain's vit-pytorch:
https://github.com/lucidrains/vit-pytorch/blob/b3e90a265284ba4df00e19fe7a1fd97ba3e3c113/vit_pytorch/simple_vit.py

Paper: https://arxiv.org/abs/2205.01580

This is compatible with both a non-B-cos SimpleViT and a B-cos SimpleViT,
provided that the correct arguments are passed.
"""
import sys                                                                                    
sys.path.insert(0, r'/BS/dnn_interpretablity_robustness_representation_learning/work/my_projects/bcos_dino')
import math

import numpy as np
from collections import OrderedDict
from typing import Any, Callable, List, Tuple, Union

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch.autograd import Variable

# other bcos modules
from bcos.modules.common import DetachableModule
from bcos.modules.bcosconv2d import *
from bcos.modules.bcoslinear import *
from bcos.modules.common import *
from bcos.modules.logitlayer import *
from bcos.modules.norms import *

from utils import trunc_normal_

# test bcos models (sanity check)
def test_bcos_model(test_model):

    # setup model for evaluation mode
    test_model.eval()
    for mod in test_model.modules():
        if hasattr(mod, "explanation_model"):
            mod.explanation_mode(True)
        
        if hasattr(mod, "set_explanation_mode"):
            mod.set_explanation_mode(True)

        if hasattr(mod, "detach"):
            mod.detach = True

        if hasattr(mod, "detach_var"):
            mod.detach_var = True

    # setting up the remaining test
    test_input = torch.randn((1, 3, 224, 224))
    im_var = Variable(test_input, requires_grad = True)

    hooks = []
    model = nn.Sequential(test_model)
    model.eval()

    def save_input(layer, input, output):
        x = input[0]
        x.retain_grad()
        layer.saved = x
    hooks.append(model[0].transformer.register_forward_hook(save_input))

    tgt = np.random.randint(1000)
    #with model.explanation_mode():
    if True:
        out = model(im_var)[0, tgt]
        out.sum().backward()

    contrib_sum = (model[0].transformer.saved*model[0].transformer.saved.grad).sum() #+ model[1].logit_bias
    print(out==contrib_sum, out.item(), contrib_sum.item())

# helpers
def exists(x: Any) -> bool:
    return x is not None


def pair(t: Any) -> Tuple[Any, Any]:
    return t if isinstance(t, tuple) else (t, t)


# classes
class PosEmbSinCos2d(nn.Module):
    def __init__(self, temperature: Union[int, float] = 10_000):
        super().__init__()
        self.temperature = temperature

    def forward(self, patches: Tensor) -> Tensor:
        h, w, dim = patches.shape[-3:]
        device = patches.device
        dtype = patches.dtype

        y, x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
        omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
        omega = 1.0 / (self.temperature**omega)

        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
        return pe.type(dtype)

# sgairola comment: drop_path(), DropPath(nn.Module) is missing in this version that exists
# in the original DINO implementation. And also we use no dropout.


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        linear_layer: Callable[..., nn.Module] = None,
        norm_layer: Callable[..., nn.Module] = None,
        act_layer: Callable[..., nn.Module] = None,
    ):
        assert exists(linear_layer), "Provide a linear layer class!"
        assert exists(norm_layer), "Provide a norm layer (compatible with LN) class!"
        assert exists(act_layer), "Provide a activation layer class!"

        super().__init__()
        self.net = nn.Sequential(
            OrderedDict(
                norm=norm_layer(dim),
                linear1=linear_layer(dim, hidden_dim),
                act=act_layer(),
                linear2=linear_layer(hidden_dim, dim),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(DetachableModule):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        linear_layer: Callable[..., nn.Module] = None,
        norm_layer: Callable[..., nn.Module] = None,
        pos_info: Tuple[str, int] = None,  # type of positional information and number of tokens,
        to_out: Callable[..., nn.Module] = None,
    ):
        assert exists(linear_layer), "Provide a linear layer class!"
        assert exists(norm_layer), "Provide a norm layer (compatible with LN) class!"

        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = norm_layer(dim)
        if not pos_info[0] == "pos_embed":
            self.pos_info = pos_info[0]
            self.register_attention_biases(pos_info[1])
        else:
            self.pos_info = None
            self.attention_biases = None

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        if to_out is None:
            self.to_out = linear_layer(inner_dim, dim, bias=False)
        else:
            self.to_out = to_out(inner_dim, dim, bias=False)

    def register_attention_biases(self, n):
        # points = list(itertools.product(range(resolution), range(resolution)))
        self.attention_biases = torch.nn.Parameter(torch.zeros(1, self.heads, n, n))

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        if self.detach:  # detach dynamic linear weights
            q = q.detach()
            k = k.detach()
            # these are used for dynamic linear w (`attn` below)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if self.pos_info is None:
            attn = self.attend(dots)
        elif self.pos_info == "add_att":
            dots = dots + self.attention_biases
            attn = self.attend(dots)
        elif self.pos_info == "mul_att":
            attn = self.attend(dots) * self.attend(self.attention_biases)
        elif self.pos_info == "mul_att_normed":
            attn = self.attend(dots) * self.attend(self.attention_biases)
            attn = attn / attn.sum(-1, keepdim=True)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out), attn


class Encoder(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        linear_layer: Callable[..., nn.Module] = None,
        norm_layer: Callable[..., nn.Module] = None,
        act_layer: Callable[..., nn.Module] = None,
        pos_info: Tuple[str, int] = None,  # type of positional information and number of tokens
        to_out: Callable[..., nn.Module] = None,
    ):
        assert exists(linear_layer), "Provide a linear layer class!"
        assert exists(norm_layer), "Provide a norm layer (compatible with LN) class!"
        assert exists(act_layer), "Provide a activation layer class!"

        super().__init__()

        self.attn = Attention(
            dim,
            heads=heads,
            dim_head=dim_head,
            linear_layer=linear_layer,
            norm_layer=norm_layer,
            pos_info=pos_info,
            to_out=to_out,
        )

        self.ff = FeedForward(
            dim,
            mlp_dim,
            linear_layer=linear_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

    def forward(self, x: Tensor, return_attention=False) -> Tensor:
        if return_attention:
            return self.attn(x)[1]

        # pick attention first dim only, since 2nd dim is attn
        x = self.attn(x)[0] + x
        x = self.ff(x) + x
        return x


class Transformer(nn.Sequential):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        linear_layer: Callable[..., nn.Module] = None,
        norm_layer: Callable[..., nn.Module] = None,
        act_layer: Callable[..., nn.Module] = None,
        pos_info: Tuple[str, int] = None,  # type of positional information and number of tokens
        to_out: Callable[..., nn.Module] = None,
    ):
        assert exists(linear_layer), "Provide a linear layer class!"
        assert exists(norm_layer), "Provide a norm layer (compatible with LN) class!"
        assert exists(act_layer), "Provide a activation layer class!"

        layers_odict = OrderedDict()
        for i in range(depth):
            layers_odict[f"encoder_{i}"] = Encoder(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                linear_layer=linear_layer,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pos_info=pos_info,
                to_out=to_out,
            )
        super().__init__(layers_odict)

class SGModifiedSimpleViT(nn.Module):
    def __init__(
        self,
        *,
        image_size: Union[int, Tuple[int, int]],
        patch_size: Union[int, Tuple[int, int]],
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int = 6,
        linear_layer: Callable[..., nn.Module] = None,
        norm_layer: Callable[..., nn.Module] = None,
        act_layer: Callable[..., nn.Module] = None,
        norm2d_layer: Callable[..., nn.Module] = None,
        conv2d_layer: Callable[..., nn.Module] = None,
        conv_stem: List[int] = None,  # Output channels for each layer of conv stem
        pos_info: str = "pos_embed", # Donot change this, as it is replaced with learned emb.
        to_out: Callable[..., nn.Module] = None,
        use_cls_token = False,
        pool_tokens_flag = True, # take mean of tokens
        **kwargs,
    ):
        super().__init__()
        _ = kwargs  # Ignore additional experiment parameters...
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert exists(linear_layer), "Provide a linear layer class!"
        assert exists(norm_layer), "Provide a norm layer (compatible with LN) class!"
        assert exists(act_layer), "Provide a activation layer class!"

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        # embedding dim added by sgairola
        self.embed_dim = embed_dim = dim
        self.use_cls_token = use_cls_token
        self.pool_tokens_flag = pool_tokens_flag
        self.input_channels = channels
        self.image_size = image_size
        self.patch_size = patch_size

        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = (
            (channels if conv_stem is None else conv_stem[-1])
            * patch_height
            * patch_width
        )
        stem = (
            dict()
            if conv_stem is None
            else dict(
                conv_stem=make_conv_stem(
                    channels, conv_stem, conv2d_layer, norm2d_layer, act_layer
                )
            )
        )
        self.to_patch_embedding = nn.Sequential(
            OrderedDict(
                **stem,
                rearrage=Rearrange(
                    "b c (h p1) (w p2) -> b h w (p1 p2 c)",
                    p1=patch_height,
                    p2=patch_width,
                ),
                linear=linear_layer(self.patch_dim, dim),
            )
        )

        if pos_info == "pos_embed":
            #self.positional_embedding = PosEmbSinCos2d()
            # replacing this to learned embedding as done in the original DINO
            # implementation.
            if self.use_cls_token:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
            else:
                self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        else:
            self.positional_embedding = lambda x: 0

         # we should also account for dropout and stochastic path dropping as DINO 
         # implementation. ignored for now.

        dim_head = dim // heads
        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            linear_layer=linear_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            pos_info=(pos_info, self.num_patches),
            to_out=to_out,
        )

        self.to_latent = nn.Identity()
        # classifier head
        self.linear_head = nn.Sequential(
            OrderedDict(
                norm=norm_layer(dim),
                linear=linear_layer(dim, num_classes),
            )
        ) if num_classes > 0 else nn.Identity()

        # DINO ViT also does some standard initializing of the layers.
        # skipping for now.

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if self.use_cls_token:
            npatch -= 1
            N -= 1
        if npatch == N and w == h:
            return self.pos_embed
        
        if self.use_cls_token:
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
        else:
            patch_pos_embed = self.pos_embed
        dim = x.shape[-1]
        
        # hardcoding this step for image_size = 14, patch_size = 1
        if self.patch_size == 1 and self.image_size in [12, 14]:
            patch_size = 16
        else:
            patch_size = self.patch_size
        w0 = w // patch_size
        h0 = h // patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        if self.use_cls_token:
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        return patch_pos_embed

    # sgairola adding this similar to DINO ViT.
    def prepare_tokens(self, x):
        # handling if bcos net
        B, nc, w, h  = x.shape
        if self.input_channels > 3 and nc < self.input_channels:
            x = torch.cat([x, 1-x], dim=1)
        B, nc, w, h  = x.shape
        x = self.to_patch_embedding(x) # patch linear embedding
        x = rearrange(x, "b ... d -> b (...) d")

        # add the [CLS] token to the embed patch tokens
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add the positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        # return the prepared tokens
        return x


    def forward(self, img):
        # the 3 lines below from moritz's original implementation.
        #x = self.to_patch_embedding(img)
        #pe = self.positional_embedding(x)
        #x = rearrange(x, "b ... d -> b (...) d") + pe

        x = self.prepare_tokens(img)
        x = self.transformer(x)
        if self.use_cls_token:
            if self.pool_tokens_flag:
                x = x[:, 0]
            else:
                x = x[:, 1:]
        else:
            if self.pool_tokens_flag:
                x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)
    
    # adding extra code here similar to the DINO VisionTransformer class
    def get_last_selfattention(self, x):
        #x = self.to_patch_embedding(x)
        #pe = self.positional_embedding(x)
        #x = rearrange(x, "b ... d -> b (...) d") + pe
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.transformer):
            if i < len(self.transformer) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        #x = self.to_patch_embedding(x)
        #pe = self.positional_embedding(x)
        #x = rearrange(x, "b ... d -> b (...) d") + pe
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.transformer):
            x = blk(x)
            if len(self.transformer) - i <= n:
                output.append(x.mean(dim=1))
                #output.append(x)
        return output


def make_conv_stem(
    in_channels: int,
    out_channels: List[int],
    conv2d_layer: Callable[..., nn.Module] = None,
    norm2d_layer: Callable[..., nn.Module] = None,
    act_layer: Callable[..., nn.Module] = None,
):
    """
    Following the conv stem design in Early Convolutions Help Transformers See Better (Xiao et al.)
    """
    model = []
    for outc in out_channels:
        conv = conv2d_layer(
            in_channels,
            outc,
            kernel_size=3,
            stride=(2 if outc > in_channels else 1),
            padding=1,
        )
        in_channels = outc
        norm = norm2d_layer(in_channels)
        act = act_layer()
        model += [conv, norm, act]
    # adaptive pooling layer added by sgairola
    # average pooling with size of 2
    #model += [nn.AdaptiveAvgPool2d((7, 7))]
    return nn.Sequential(*model)

CONFIGS = dict(                                                                           
        num_classes=1_000,                                                                
        norm_layer=NoBias(DetachableLayerNorm),                                           
        act_layer=nn.Identity,                                                            
        channels=6,                                                                       
        norm2d_layer=NoBias(DetachableGNLayerNorm2d),                                     
        linear_layer=BcosLinear,                                                          
        conv2d_layer=BcosConv2d,                                                          
    )


def vitc_ti_patch1_14(my_configs=CONFIGS):
    #kwargs.setdefault("num_classes", 1_000)
    return SGModifiedSimpleViT(
        image_size=14,
        patch_size=1,
        depth=12
        - 1,  # Early convs. help transformers see better: reduce depth to account for conv stem for fairness
        dim=384 // 2,
        heads=6 // 2,
        mlp_dim=1536 // 2,
        conv_stem=[24, 48, 96, 192],
		**my_configs,
    )


def vitc_s_patch1_14(my_configs=CONFIGS):
    #kwargs.setdefault("num_classes", 1_000)
    return SGModifiedSimpleViT(
        image_size=14,
        patch_size=1,
        depth=12
        - 1,  # Early convs. help transformers see better: reduce depth to account for conv stem for fairness
        dim=384,
        heads=6,
        mlp_dim=1536,
        # changing this to powers of 2
        conv_stem=[48, 96, 192, 384],
		**my_configs,
    )


def vitc_b_patch1_14(my_configs=CONFIGS):
    #kwargs.setdefault("num_classes", 1_000)
    return SGModifiedSimpleViT(
        image_size=14,
        patch_size=1,
        depth=12
        - 1,  # Early convs. help transformers see better: reduce depth to account for conv stem for fairness
        dim=384 * 2,
        heads=6 * 2,
        mlp_dim=1536 * 2,
        conv_stem=[64, 128, 128, 256, 256, 512],
		**my_configs,
    )

# smaller dim config

def vitc_ti_patch1_12(my_configs=CONFIGS):
    #kwargs.setdefault("num_classes", 1_000)
    return SGModifiedSimpleViT(
        image_size=12,
        patch_size=1,
        depth=12
        - 1,  # Early convs. help transformers see better: reduce depth to account for conv stem for fairness
        dim=384 // 2,
        heads=6 // 2,
        mlp_dim=1536 // 2,
        conv_stem=[24, 48, 96, 192],
		**my_configs,
    )


def vitc_s_patch1_12(my_configs=CONFIGS):
    #kwargs.setdefault("num_classes", 1_000)
    return SGModifiedSimpleViT(
        image_size=12,
        patch_size=1,
        depth=12
        - 1,  # Early convs. help transformers see better: reduce depth to account for conv stem for fairness
        dim=384,
        heads=6,
        mlp_dim=1536,
        # changing this to powers of 2
        conv_stem=[48, 96, 192, 384],
		**my_configs,
    )


def vitc_b_patch1_12(my_configs=CONFIGS):
    #kwargs.setdefault("num_classes", 1_000)
    return SGModifiedSimpleViT(
        image_size=12,
        patch_size=1,
        depth=12
        - 1,  # Early convs. help transformers see better: reduce depth to account for conv stem for fairness
        dim=384 * 2,
        heads=6 * 2,
        mlp_dim=1536 * 2,
        conv_stem=[64, 128, 128, 256, 256, 512],
		**my_configs,
    )


def simple_vit_s_patch16_224(my_configs=CONFIGS):
    #kwargs.setdefault("num_classes", 1_000)
    return SGModifiedSimpleViT(
        image_size=224,
        patch_size=16,
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
		**my_configs,
    )
def simple_vit_s_patch16_192(my_configs=CONFIGS):
    #kwargs.setdefault("num_classes", 1_000)
    return SGModifiedSimpleViT(
        image_size=192,
        patch_size=16,
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
		**my_configs,
    )


if __name__ == "__main__":                                                                    
                                                                                              
    #test_model = vitc_b_patch1_14()                                                           
    test_model =vitc_s_patch1_12(my_configs=CONFIGS)
    test_bcos_model(test_model)                                                               
    print(test_model)
    exit(0) 
##########DINO LIKE MODEL CONFIGS################
