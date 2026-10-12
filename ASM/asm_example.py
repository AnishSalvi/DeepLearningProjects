#installs
!pip install --quiet monai transformers
#imports
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Type, List, Dict
import transformers
from transformers import CLIPTextConfig, CLIPTokenizer, CLIPTextModel, AutoTokenizer
import monai
import numpy as np
import math
#Image Encoder

# Copyright (c) ImageRx and Anish Salvi.
# All rights reserved.

class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (64, 64, 64),
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        in_chans: int = 1,
        embed_dim: int = 768, #embed_dim / num_heads = img_size? at least 64, not depth
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()

        #store
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.out_chans = out_chans

        self.patch_embed = PatchEmbed(
            kernel_size = patch_size,
            stride = patch_size,
            in_chans = in_chans,
            embed_dim = embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2], embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv3d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm3d(out_chans),
            nn.Conv3d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm3d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x.permute(0, 4, 1, 2, 3))

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionViT(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W, D = x.shape[1], x.shape[2], x.shape[3]
            x, pad_hwd = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hwd, (H, W, D))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class AttentionViT(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            self.rel_pos_d = nn.Parameter(torch.zeros(2 * input_size[2] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, D, _ = x.shape
        # qkv with shape (3, B, nHead, H * W * D, C)
        qkv = self.qkv(x).reshape(B, H * W * D, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W * D, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W * D, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, self.rel_pos_d, (H, W, D), (H, W, D))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, D, -1).permute(0, 2, 3, 4, 1, 5).reshape(B, H, W, D, -1)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, C, H, W, D].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, window_size, C].
        (Hp, Wp, Dp): padded height and width before partition
    """
    B, C, H, W, D = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    pad_d = (window_size - D % window_size) % window_size
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
    Hp, Wp, Dp = H + pad_h, W + pad_w, D + pad_d

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, Dp // window_size, window_size, C)
    windows = x.contiguous().view(-1, window_size, window_size, window_size, C) #permute removed
    return windows, (Hp, Wp, Dp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hwd: Tuple[int, int, int], hwd: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, window_size, C].
        window_size (int): window size.
        pad_hwd (Tuple): padded height and width (Hp, Wp, Dp).
        hwd (Tuple): original height and width (H, W, D) before padding.

    Returns:
        x: unpartitioned sequences with [B, C, H, W, D].
    """
    Hp, Wp, Dp = pad_hwd
    H, W, D = hwd
    B = windows.shape[0] // (Hp * Wp * Dp // window_size // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, Dp // window_size, window_size, window_size, window_size, -1)
    x = x.contiguous().view(B, Hp, Wp, Dp, -1) #permute removed

    if Hp > H or Wp > W or Dp > D:
        x = x[:, :H, :W, :D, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    rel_pos_d: torch.Tensor,
    q_size: Tuple[int, int, int],
    k_size: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w * q_d, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        rel_pos_d (Tensor): relative position embeddings (Ld, C) for depth axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w, q_d).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w, k_d).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w, q_d = q_size
    k_h, k_w, k_d = k_size

    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    Rd = get_rel_pos(q_d, k_d, rel_pos_d)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, q_d, dim)
    rel_h = torch.einsum("bhwdc,hkc->bhwdk", r_q, Rh)
    rel_w = torch.einsum("bhwdc,wkc->bhwdk", r_q, Rw)
    rel_d = torch.einsum("bhwdc,dkc->bhwdk", r_q, Rd)

    attn = (
        attn.view(B, q_h, q_w, q_d, k_h, k_w, k_d)
        + rel_h[:, :, :, :, :, None] + rel_w[:, :, :, :, None, :] + rel_d[:, :, :, None, :, :]
        ).view(B, q_h * q_w * q_d, k_h * k_w * k_d)

    return attn

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int, int] = (16, 16, 16),
        stride: Tuple[int, int, int] = (16, 16, 16),
        padding: Tuple[int, int, int] = (0, 0, 0),
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W D -> B H W D C
        x = x.permute(0, 2, 3, 4, 1)
        return x
#Prompt Encoder

# Copyright (c) ImageRx and Anish Salvi.
# All rights reserved.

def divide(test_tup1, test_tup2):
  return tuple(ele1 // ele2 for ele1, ele2 in zip(test_tup1, test_tup2))

def get_down_scaling(kernel_stride_size, mask_in_chans, embed_dim):
      max_downscaling = nn.Sequential()
      #iterate
      for i in range(len(kernel_stride_size)):
        max_downscaling.add_module("conv_" + str(i), nn.Conv3d(mask_in_chans[i], mask_in_chans[i + 1],
                                                           kernel_size = kernel_stride_size[i], stride = kernel_stride_size[i]))
        max_downscaling.add_module("norm_" + str(i), LayerNorm3d(mask_in_chans[i + 1]))
        max_downscaling.add_module("act_" + str(i), torch.nn.GELU())
      max_downscaling.add_module("embed_" + str(0), torch.nn.Conv3d(mask_in_chans[i + 1], embed_dim, kernel_size=1))
      #return
      return max_downscaling


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        input_image_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        mask_in_chans: List,
        kernel_stride_size: List,
        num_attention_heads: int,
        num_hidden_layers: int,
        projection_dim: int,
        intermediate_size: int,
        max_position_embeddings: int,
        activation: Type[nn.Module] = nn.GELU,
        tokenizer: Optional[AutoTokenizer] = None
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W, D).
          patch_size (tuple(int, int, int)): The spatial size of the
            image embedding, as (H, W, D).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.image_embedding_size = divide(self.input_image_size, patch_size)

        #might need deeper network?
        self.mask_downscaling = get_down_scaling(kernel_stride_size, mask_in_chans, embed_dim)
        self.no_mask_embed = nn.Embedding(1, embed_dim)

        #init
        path = "openai/clip-vit-base-patch32"
        #get config
        config = CLIPTextConfig.from_pretrained(path)
        #if
        if tokenizer is None:
          #get tokenizer
          self.tokenizer = CLIPTokenizer.from_pretrained(path)
          #set
          config.bos_token_id = self.tokenizer.bos_token_id
          config.eos_token_id = self.tokenizer.eos_token_id
        else:
          #get tokenizer
          self.tokenizer = tokenizer
          #set
          config.bos_token_id = self.tokenizer.bos_token_id
          config.eos_token_id = self.tokenizer.eos_token_id
        #set
        config.num_attention_heads = num_attention_heads
        config.num_hidden_layers = num_hidden_layers
        config.projection_dim = projection_dim
        config.intermediate_size = intermediate_size
        config.max_position_embeddings = max_position_embeddings
        #set hidden state
        config.hidden_size = embed_dim
        #model
        self.text_model = CLIPTextModel(config)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)x(embedding_d)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 3), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 3)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _embed_text(self, text):
        """Embeds text inputs."""
        inputs = self.tokenizer(text, padding=True, return_tensors="pt").to(self._get_device())
        outputs = self.text_model(**inputs)
        return outputs.pooler_output[:, None, :]

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        text: Optional[List]
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points[0] is not None:
          return points[0].shape[0]
        elif boxes is not None:
          return boxes.shape[0]
        elif text is not None:
          return len(text)
        elif masks is not None:
          return masks.shape[0]
        else:
          return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        text: Optional[List],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed
          text  (list): text to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)x(embed_D)
        """
        bs = self._get_batch_size(points, boxes, masks, text)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points[0] is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if text is not None:
            text_embeddings = self._embed_text(text)
            sparse_embeddings = torch.cat([sparse_embeddings, text_embeddings], dim=1)
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1], self.image_embedding_size[2]
                )

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w, d = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w, d), device=device, dtype=torch.float32)
        #embed
        x_embed = grid.cumsum(dim=0) - 0.5
        y_embed = grid.cumsum(dim=1) - 0.5
        z_embed = grid.cumsum(dim=2) - 0.5
        #norm
        x_embed = x_embed / h
        y_embed = y_embed / w
        z_embed = z_embed / d
        #encode
        pe = self._pe_encoding(torch.stack([x_embed, y_embed, z_embed], dim=-1))
        return pe.permute(3, 0, 1, 2)  # C x H x W x D

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[0]
        coords[:, :, 1] = coords[:, :, 1] / image_size[1]
        coords[:, :, 2] = coords[:, :, 2] / image_size[2]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
    
#Transformer

# Copyright (c) ImageRx and Anish Salvi.
# All rights reserved.

class TwoWayTransformerMLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = TwoWayTransformerAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w x d for any h and w and d.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        #bs, c, h, w, d = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = TwoWayTransformerAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = TwoWayTransformerAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = TwoWayTransformerMLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = TwoWayTransformerAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class TwoWayTransformerAttention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
    
#Mask Decoder

# Copyright (c) ImageRx and Anish Salvi.
# All rights reserved.

def get_output_scaling(kernel_stride_size, transformer_dim):
      output_upscaling = nn.Sequential()
      #iterate
      for i in range(len(kernel_stride_size)):
        output_upscaling.add_module("conv_" + str(i), nn.ConvTranspose3d(transformer_dim[i], transformer_dim[i+1],
                                                                         kernel_size = kernel_stride_size[i], stride = kernel_stride_size[i]))
        if len(kernel_stride_size) - 1 == i:
          output_upscaling.add_module("act_" + str(i), torch.nn.GELU())
          continue
        else:
          output_upscaling.add_module("norm_" + str(i), LayerNorm3d(transformer_dim[i + 1]))
          output_upscaling.add_module("act_" + str(i), torch.nn.GELU())
      #return
      return output_upscaling

def get_MLP(start_dim, end_dim, num_mask_tokens, num_layers):
  output_hypernetworks_mlps = nn.ModuleList([
      MLP(start_dim, start_dim, end_dim, num_layers)
      for i in range(num_mask_tokens)
      ])
  return output_hypernetworks_mlps


class MaskDecoder(nn.Module):
    def __init__(
        self,
        kernel_stride_size: List,
        transformer_dim: List,
        patch_size: Tuple[int, int, int],
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        num_layers: int = 3
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          patch_size Tuple(int, int, int): the patch size used in the encoder
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim[0])
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim[0])
        self.patch_size = patch_size

        self.output_upscaling = get_output_scaling(kernel_stride_size, transformer_dim)
        self.output_hypernetworks_mlps = get_MLP(transformer_dim[0], transformer_dim[-1], self.num_mask_tokens, num_layers)

        self.iou_prediction_head = MLP(
            transformer_dim[0], iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w, d = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w, d)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        #iterate
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        #b, c, h, w = upscaled_embedding.shape this is likely if the upscaled embedding does not have the same size if altering upsampling
        b, c, h, w, d = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w * d)).view(b, -1, h, w, d)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
    
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

class AnishSalviModel(nn.Module):

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (64, 64, 64),
        patch_size: Tuple[int, int, int] = (8, 8, 8),
        in_chans: int = 1,
        image_embed_dim: int = 768, #has to be 768?
        image_encoder_depth: int = 12,
        image_encoder_num_heads: int = 12,
        image_encoder_mlp_ratio: int = 4,
        image_encoder_out_chans: int = 512, #same?
        prompt_encoder_embed_dim: int = 512, #same?
        prompt_encoder_mask_in_chans: List = [1, 3, 5],
        prompt_encoder_kernel_stride_size: List = [(2, 2, 2), (4, 4, 4)],
        prompt_encoder_num_attention_heads: int = 4,
        prompt_encoder_num_hidden_layers: int = 8,
        prompt_encoder_projection_dim: int = 64,
        prompt_encoder_intermediate_size: int = 64,
        prompt_encoder_max_position_embeddings: int = 16,
        transformer_depth: int = 16,
        transformer_embedding_dim: int = 512, #same?
        transformer_num_heads: int = 16,
        transformer_mlp_dim: int = 8,
        mask_decoder_kernel_stride_size: List = [(4, 4, 4), (2, 2, 2)],
        mask_decoder_transformer_dim: List = [512, 512, 512],
        mask_decoder_num_multimask_outputs: int = 3,
        mask_decoder_num_layers: int = 3,
        tokenizer: Optional[AutoTokenizer] = None
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          transformer (TwoWayTransformer): Processes embeddings within mask_decoder
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
        """
        super().__init__()
        #image encoder
        self.image_encoder = ImageEncoderViT(img_size, patch_size, in_chans, image_embed_dim,
                                image_encoder_depth, image_encoder_num_heads, image_encoder_mlp_ratio, image_encoder_out_chans)

        #prompt encoder
        self.prompt_encoder = PromptEncoder(prompt_encoder_embed_dim,
                                            img_size, patch_size, prompt_encoder_mask_in_chans, prompt_encoder_kernel_stride_size,
                                            prompt_encoder_num_attention_heads,
                                            prompt_encoder_num_hidden_layers, prompt_encoder_projection_dim,
                                            prompt_encoder_intermediate_size, prompt_encoder_max_position_embeddings, tokenizer)
        #transformer
        transformer = TwoWayTransformer(transformer_depth, transformer_embedding_dim,
                                        transformer_num_heads, transformer_mlp_dim)
        #mask decoder
        self.mask_decoder = MaskDecoder(mask_decoder_kernel_stride_size, mask_decoder_transformer_dim, patch_size, transformer,
                                        mask_decoder_num_multimask_outputs, mask_decoder_num_layers)


    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using ASMpredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in CxHxWxD format,
                already transformed for input to the model.
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx3. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx6.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxWxD.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([x["image"] for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            #encode embeddings
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=(image_record["point_coords"], image_record["point_labels"]),
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
                text=image_record.get("text", None)
            )
            #decode embeddings
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            outputs.append(
                {
                    "iou_predictions": iou_predictions,
                    "logits": low_res_masks,
                }
            )
        return outputs
    
#dice focal loss multilabel
class DiceFocalIOULoss:
  def __init__(self, include_background = True, to_onehot_y = False, thresh = 0.5,
               gamma = 2.0, focal_weight = None, lambda_dice = 1.0, lambda_focal = 1.0, lambda_iou = 1.0):
    #dice focal loss
    self.loss_fn1 = monai.losses.DiceFocalLoss(include_background = include_background, to_onehot_y = to_onehot_y,
                                               sigmoid = True, softmax = False,
                                               gamma = gamma, focal_weight = focal_weight,
                                                lambda_dice = lambda_dice, lambda_focal = lambda_focal)
    self.fn2 = monai.metrics.MeanIoU(reduction = 'none')
    #l1 loss
    self.loss_fn3 = torch.nn.MSELoss()
    #set
    self.thresh = thresh
    self.lambda_iou = lambda_iou

  def __call__(self, batched_output, batched_input):
    #init
    loss_fn1 = 0
    loss_fn3 = 0
    #iterate y_pred, y_true
    for i, (output, input) in enumerate(zip(batched_output, batched_input)):
      #if sigmoid else softmax
      y_pred = torch.swapaxes(output['logits'], 0, 1)
      #y_true
      y_true = torch.swapaxes(input['mask_inputs'], 0, 1)
      #dice focal loss
      loss_fn1 += self.loss_fn1(y_pred, y_true)
      #proxy for iou loss
      y_pred = torch.sigmoid(y_pred)
      iou_y_true = self.fn2(torch.where(y_pred > self.thresh, 1, 0), y_true)
      #apply
      loss_fn3 += self.loss_fn3(torch.sigmoid(output['iou_predictions']), iou_y_true.T)

    #divide
    return (loss_fn1 + self.lambda_iou * loss_fn3) / len(batched_input)

#init
if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

#init
model = AnishSalviModel()
criterion = DiceFocalIOULoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5, weight_decay = 0.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 15, gamma=0.7, verbose=True)

#send
model.to(device)

#input data

#boxes
boxes = torch.Tensor([
    [16, 16, 16, 36, 36, 36],
    [0, 0, 0, 63, 63, 63],
     [20, 20, 20, 30, 30, 30]])

#get label
label = torch.zeros((64, 64, 64))
label[16:36,16:36,16:36] = 1
label[20:30,20:30,20:30] = 2

#point coords
point_coords = torch.Tensor([[[18, 18, 18], [33, 33, 33]], [[4, 4, 4], [60, 60, 60]], [[24, 24, 24], [26, 26, 26]]])
#point_coords = torch.Tensor([[[4, 4, 4], [60, 60, 60]], [[18, 18, 18], [33, 33, 33]], [[24, 24, 24], [26, 26, 26]]])

#point labels
point_labels = torch.Tensor([[1, 1], [1, 1], [1, 1]]) #0 if outside and -1 if background; in multiclass likely 0s and 1s

#mask inputs
mask_inputs = torch.zeros((3, 1, 64, 64, 64))
mask_inputs[0, 0, ] = torch.where(label >= 1, 1, 0)
mask_inputs[1, 0, ] = torch.where(label == 0, 1, 0)
mask_inputs[2, 0, ] = torch.where(label == 2, 1, 0)

#image
image = (label - label.mean()) / label.std()
#image add channel
image = torch.unsqueeze(image, 0)

#text
text = ['lungs', 'aorta', 'brain']

#get dict
dict0 = {
    'image': image.to(device),
    'point_coords': None, #, point_coords.to(device),
    'point_labels': None, #point_labels.to(device),
    'boxes': None, #boxes.to(device),
    'mask_inputs': mask_inputs.to(device),
    'text': text
    }

#get batch
batched_input = [dict0] #[dict0, dict0]

#iterate
for i in range(500):
  #zero optimizer
  optimizer.zero_grad()
  #cast
  #with torch.amp.autocast(device, torch.float16):
  #forward pass
  batched_output = model(batched_input, False)
  #compute loss
  loss = criterion(batched_output, batched_input)
  #check backwards
  loss.backward()
  #step
  optimizer.step()
  #step
  scheduler.step()
  #print
  print(loss.item())
    
#quick output
!pip install --quiet SimpleITK
import SimpleITK as sitk

image = sitk.GetImageFromArray(np.where(torch.sigmoid(batched_output[0]['logits'][1, 0]).cpu().detach().numpy() > 0.5, 1, 0))
image = sitk.Cast(image, sitk.sitkUInt32)
sitk.WriteImage(image, '/content/temp.nii.gz')

image = sitk.GetImageFromArray(mask_inputs[1, 0].cpu().detach().numpy())
image = sitk.Cast(image, sitk.sitkUInt32)
sitk.WriteImage(image, '/content/temp_mask.nii.gz')