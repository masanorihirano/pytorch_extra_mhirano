import math
import warnings
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn

__all__ = ["DotProductAttention", "SelfAttention", "SelfMultiheadAttention"]


class DotProductAttention(nn.Module):
    """DotProductAttention.

    .. math::

        \mathrm{DotProductAttention}(Q, K, V) &=& \mathrm{softmax}(qk^T) v

        q &=& QW_1 + b_1

        k &=& KW_2 + b_2

        v &=& VW_3 + b_3

    Args:
        qdim: dimension of the model, i.e., dimension of Q
        hidden_dim: dimension of hidden layer, i.e., dimension of q, k, v. Default: 512
        output_dim: dimension of output layer, i.e., dimension of output. Default: None
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        transform: q = Q, k = K, v = V if it is False. Default: True
        bias: add bias as module parameter. Default: True.
        same_embd: W1 = W2 = W3, b1 = b2 = b3 if it is True. Default: True
        add_bias_kv: add bias to the key and value sequences at dim=0.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::
        >>> attn = DotProductAttention(query_dim)
        >>> attn_output, attn_output_weights = attn(query, key, value)
    """

    def __init__(
        self,
        qdim: int,
        output_dim: Optional[int] = None,
        dropout: float = 0.0,
        transform: bool = True,
        bias: bool = True,
        same_embd: bool = True,
        add_bias_kv: Optional[bool] = None,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True,
        scaled: bool = False,
    ) -> None:
        super(DotProductAttention, self).__init__()
        self.qdim: int = qdim
        self.transform: bool = transform
        self.bias: bool = bias
        self.same_embd: bool = same_embd
        self.kdim: int = kdim if kdim is not None else self.qdim
        self.vdim: int = vdim if vdim is not None else self.qdim
        self.output_dim: int = output_dim if output_dim is not None else self.vdim
        if self.same_embd and (self.qdim != self.kdim or self.qdim != self.vdim):
            raise AssertionError(
                "qdim, kdim, vdim should be the same dimensions if same_embd is True"
            )
        self.add_bias_kv: bool = add_bias_kv if add_bias_kv is not None else self.bias
        if self.same_embd and (self.bias != self.add_bias_kv):
            raise AssertionError(
                "bias and add_bias_kv should be the same if same_embd is True"
            )
        self.batch_first: bool = batch_first
        self.scaled: bool = scaled

        self.fc_q: nn.Module = nn.Linear(self.qdim, self.output_dim, bias=bias)
        self.fc_k: nn.Module
        self.fc_v: nn.Module
        if self.same_embd:
            self.fc_k = self.fc_q
            self.fc_v = self.fc_k
        else:
            self.fc_k = nn.Linear(self.kdim, self.output_dim, bias=self.add_bias_kv)
            self.fc_v = nn.Linear(self.vdim, self.output_dim, bias=self.add_bias_kv)
        self.dropout: nn.Module = nn.Dropout(p=dropout)
        self.softmax: nn.Module = nn.Softmax(dim=2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_padding_mask is not None:
            warnings.warn(
                "'key_padding_mask' in 'DotProductAttention' is currently an experimental version."
                "When you use this, please check if this is working correctly or not very carefully."
            )
        if not self.batch_first:
            query = torch.transpose(query, 0, 1)
            key = torch.transpose(key, 0, 1)
            value = torch.transpose(value, 0, 1)

        bsz, tgt_len, _ = query.size()

        q = self.fc_q(query)
        q = self.dropout(q)
        k = self.fc_k(key)
        k = self.dropout(k)
        v = self.fc_v(value)
        v = self.dropout(v)

        if k.size() != v.size():
            raise AssertionError("The sizes of key and value should be the same.")
        src_len = k.size(1)
        if key_padding_mask is not None:
            if key_padding_mask.size(0) != bsz:
                raise AssertionError(
                    "The first dimension of kay padding mask size must be the same as batch size"
                )
            if key_padding_mask.size(1) != src_len:
                raise AssertionError(
                    "The second dimension of key padding mask size must be the same as source length"
                )

        a = torch.bmm(q, torch.transpose(k, 1, 2))
        if self.scaled:
            a /= math.sqrt(self.output_dim)

        if attn_mask is not None:
            a += attn_mask

        if key_padding_mask is not None:
            a = a.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))

        attn = self.softmax(a)
        output = torch.bmm(attn, v)
        if not self.batch_first:
            output = torch.transpose(output, 0, 1)
        return output, attn

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask


class SelfAttention:
    def __init__(
        self,
        qdim: int,
        dropout: float = 0.0,
        transform: bool = True,
        bias: bool = True,
        same_embd: bool = True,
        add_bias_kv: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True,
        scaled: bool = False,
    ) -> None:
        self.attn = DotProductAttention(
            qdim=qdim,
            output_dim=qdim,
            dropout=dropout,
            transform=transform,
            bias=bias,
            same_embd=same_embd,
            add_bias_kv=add_bias_kv,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            scaled=scaled,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.attn.forward(
            inputs,
            inputs,
            inputs,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )


class SelfMultiheadAttention:
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.attn.forward(
            query=inputs,
            key=inputs,
            value=inputs,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
        )
