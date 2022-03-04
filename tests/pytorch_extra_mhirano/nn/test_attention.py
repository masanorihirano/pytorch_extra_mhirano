import pytest
import torch
from torch.testing import assert_close

from pytorch_extra_mhirano.nn.attention import DotProductAttention
from pytorch_extra_mhirano.nn.attention import SelfAttention
from pytorch_extra_mhirano.nn.attention import SelfMultiheadAttention


class TestDotProductionAttention:
    def test__init__(self) -> None:
        layer = DotProductAttention(qdim=10)
        ones = torch.ones([1, 2, 10])
        result, attn = layer.forward(query=ones, key=ones, value=ones)
        assert_close(attn, torch.ones([1, 2, 2]) * 0.5)
        assert result.size() == torch.Size([1, 2, 10])

    def test__init__2(self) -> None:
        with pytest.raises(AssertionError):
            DotProductAttention(qdim=10, same_embd=True, kdim=5)
        with pytest.raises(AssertionError):
            DotProductAttention(qdim=10, same_embd=True, vdim=5)
        DotProductAttention(qdim=10, same_embd=True, bias=False, add_bias_kv=False)
        with pytest.raises(AssertionError):
            DotProductAttention(qdim=10, same_embd=True, bias=False, add_bias_kv=True)
        DotProductAttention(qdim=10, same_embd=False)

    def test__init__batch_first_false(self) -> None:
        layer = DotProductAttention(qdim=10, batch_first=False)
        ones = torch.ones([2, 1, 10])
        result, attn = layer.forward(query=ones, key=ones, value=ones)
        assert_close(attn, torch.ones([1, 2, 2]) * 0.5)
        assert result.size() == torch.Size([2, 1, 10])


class TestSelfAttention:
    def test__init__(self) -> None:
        layer = SelfAttention(qdim=10)
        ones = torch.ones([1, 2, 10])
        result, attn = layer.forward(ones)
        assert_close(attn, torch.ones([1, 2, 2]) * 0.5)
        assert result.size() == torch.Size([1, 2, 10])

    def test__init__2(self) -> None:
        with pytest.raises(AssertionError):
            SelfAttention(qdim=10, same_embd=True, kdim=5)
        with pytest.raises(AssertionError):
            SelfAttention(qdim=10, same_embd=True, vdim=5)
        SelfAttention(qdim=10, same_embd=True, bias=False, add_bias_kv=False)
        with pytest.raises(AssertionError):
            SelfAttention(qdim=10, same_embd=True, bias=False, add_bias_kv=True)
        SelfAttention(qdim=10, same_embd=False)

    def test__init__batch_first_false(self) -> None:
        layer = SelfAttention(qdim=10, batch_first=False)
        ones = torch.ones([2, 1, 10])
        result, attn = layer.forward(ones)
        assert_close(attn, torch.ones([1, 2, 2]) * 0.5)
        assert result.size() == torch.Size([2, 1, 10])


class TestSelfMultiheadAttention:
    def test__init__(self) -> None:
        layer = SelfMultiheadAttention(embed_dim=10, num_heads=2)
        ones = torch.ones([1, 2, 10])
        result, attn = layer.forward(ones)
        assert_close(attn, torch.ones([1, 2, 2]) * 0.5)
        assert result.size() == torch.Size([1, 2, 10])

    def test__init__batch_first_false(self) -> None:
        layer = SelfMultiheadAttention(embed_dim=10, num_heads=2, batch_first=False)
        ones = torch.ones([2, 1, 10])
        result, attn = layer.forward(ones)
        assert_close(attn, torch.ones([1, 2, 2]) * 0.5)
        assert result.size() == torch.Size([2, 1, 10])
