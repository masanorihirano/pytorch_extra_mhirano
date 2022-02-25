from pytorch_extra_mhirano.nn.attention import DotProductAttention


class TestDotProductionAttention:
    def test__init__(self) -> None:
        layer = DotProductAttention(qdim=10)
