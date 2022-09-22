import pytest
import torch

from pytorch_extra_mhirano.utils import ReplayBuffer


class TestReplayBuffer:
    def test__init__(self) -> None:
        rb = ReplayBuffer(max_samples=100)
        assert rb.max_samples == 100
        assert rb.torch_compatible == True
        rb = ReplayBuffer(max_samples=10, torch_compatible=False)
        assert rb.max_samples == 10
        assert rb.torch_compatible == False

    @pytest.mark.parametrize("size", [10, 100, 1000])
    @pytest.mark.parametrize("max_size", [10, 100, 1000])
    def test__len__(self, size: int, max_size: int) -> None:
        rb = ReplayBuffer(max_samples=max_size)
        data = {"data": torch.rand(size)}
        rb.batch_push(samples=data)
        assert len(rb) == min(size, max_size)

    def test_keys(self) -> None:
        rb = ReplayBuffer(max_samples=100)
        assert rb.keys() is None
        data = {"data": torch.rand(10), "data2": torch.rand(10)}
        rb.batch_push(samples=data)
        assert rb.keys() == ["data", "data2"]

    def test_append(self) -> None:
        rb = ReplayBuffer(max_samples=10)
        for i in range(20):
            data = {"data": torch.rand(1), "data2": torch.rand(1)}
            rb.append(sample=data)
            assert len(rb) == min(i + 1, 10)

    def test_can_sample(self) -> None:
        rb = ReplayBuffer(max_samples=10)
        assert rb.can_sample(batch_size=1) == False
        data = {"data": torch.rand(10), "data2": torch.rand(10)}
        rb.batch_push(samples=data)
        assert rb.can_sample(batch_size=1) == True
        assert rb.can_sample(batch_size=10) == True
        assert rb.can_sample(batch_size=11) == False

    def test_extends(self) -> None:
        rb = ReplayBuffer(max_samples=10)
        assert rb.can_sample(batch_size=1) == False
        data = [{"data": torch.rand(1), "data2": torch.rand(1)} for _ in range(100)]
        rb.extends(samples=data)
        assert rb.can_sample(batch_size=1) == True
        assert rb.can_sample(batch_size=10) == True
        assert rb.can_sample(batch_size=11) == False
        assert rb.buffer == data[-10:]

    def test_batch_push(self) -> None:
        rb = ReplayBuffer(max_samples=10)
        assert rb.can_sample(batch_size=1) == False
        data = {"data": torch.rand(100), "data2": torch.rand(100)}
        rb.batch_push(samples=data)
        assert rb.can_sample(batch_size=1) == True
        assert rb.can_sample(batch_size=10) == True
        assert rb.can_sample(batch_size=11) == False
        torch.testing.assert_close(
            torch.stack([x["data"] for x in rb.buffer]), data["data"][-10:]
        )
        torch.testing.assert_close(
            torch.stack([x["data2"] for x in rb.buffer]), data["data2"][-10:]
        )

    def test_sample(self) -> None:
        rb = ReplayBuffer(max_samples=40)
        assert rb.can_sample(batch_size=1) == False
        data = {"data": torch.rand(100), "data2": torch.rand(100)}
        rb.batch_push(samples=data)
        results = rb.sample(batch_size=10)
        assert len(results) == 10
        assert list(results[0].keys()) == ["data", "data2"]

    def test_torch_sample(self) -> None:
        rb = ReplayBuffer(max_samples=40)
        assert rb.can_sample(batch_size=1) == False
        data = {"data": torch.rand(100), "data2": torch.rand(100)}
        rb.batch_push(samples=data)
        results = rb.torch_sample(batch_size=10)
        assert list(results.keys()) == ["data", "data2"]
        assert results["data"].size() == torch.Size([10])
        assert results["data2"].size() == torch.Size([10])

        rb = ReplayBuffer(max_samples=40, torch_compatible=False)
        assert rb.can_sample(batch_size=1) == False
        data = {"data": torch.rand(100), "data2": torch.rand(100)}
        rb.batch_push(samples=data)
        with pytest.raises(RuntimeError):
            rb.torch_sample(batch_size=10)
