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
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature)
        scaled: If ``True``, this performs as scaled dot product attention

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
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
                or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
                :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
                Queries are compared against key-value pairs to produce the output.
                See "Attention Is All You Need" for more details.
            key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
                or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
                :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
                See "Attention Is All You Need" for more details.
            value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
                ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
                sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
                See "Attention Is All You Need" for more details.
            key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
                to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
                Binary and byte masks are supported.
                For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
                the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
                value will be ignored.
            need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
                Default: ``True``.
            attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
                :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
                :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
                broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
                Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
                corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
                corresponding position is not allowed to attend. For a float mask, the mask values will be added to
                the attention weight.

        Outputs:
            - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
              :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
              where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
              embedding dimension ``embed_dim``.
            - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
              returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
              :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
              :math:`S` is the source sequence length. If ``average_weights=False``, returns attention weights per
              head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.

            .. note::
                `batch_first` argument is ignored for unbatched inputs.
        """
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
        if need_weights:
            return output, attn
        else:
            return output, None

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


class SelfAttention(nn.Module):
    """Self Attention module using DotProductAttention

    Args:
        qdim: dimension of the model, i.e., dimension of Q
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        transform: q = Q, k = K, v = V if it is False. Default: True
        bias: add bias as module parameter. Default: True.
        same_embd: W1 = W2 = W3, b1 = b2 = b3 if it is True. Default: True
        add_bias_kv: add bias to the key and value sequences at dim=0.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature)
        scaled: If ``True``, this performs as scaled dot product attention
    """

    def __init__(
        self,
        qdim: int,
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
        super(SelfAttention, self).__init__()
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
        need_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.attn.forward(
            inputs,
            inputs,
            inputs,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
        )


class SelfMultiheadAttention(nn.Module):
    """Self Attention module using torch.nn.MultiheadAttention

    Args:
        embed_dim: dimension of the model, i.e., dimension of Q
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: Optional[bool] = None,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super(SelfMultiheadAttention, self).__init__()
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
