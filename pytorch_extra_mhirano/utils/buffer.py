import random
from typing import Dict
from typing import List
from typing import Optional
from typing import cast

import torch


class ReplayBuffer:
    def __init__(self, max_samples: int, torch_compatible: bool = True) -> None:
        self.max_samples: int = max_samples
        self.buffer: List[Dict] = []
        self.torch_compatible = torch_compatible

    def __len__(self) -> int:
        return len(self.buffer)

    def keys(self) -> Optional[List[str]]:
        if len(self.buffer) == 0:
            return None
        return list(self.buffer[0].keys())

    def append(self, sample: Dict) -> None:
        self.extends([sample])

    def can_sample(self, batch_size: int) -> bool:
        return batch_size <= self.__len__()

    def extends(self, samples: List[Dict]) -> None:
        keys = self.keys()
        if keys is not None:
            keys_set = set(keys)
            if sum(map(lambda sample: keys_set != set(sample.keys()), samples)) != 0:
                raise ValueError("keys in the inputs is not matched to the buffer data")
        else:
            keys = list(samples[0].keys())
        keys_: List = cast(List, keys)

        if self.torch_compatible:
            if (
                sum(
                    map(
                        lambda sample: sum(
                            map(
                                lambda key: not isinstance(sample[key], torch.Tensor),
                                keys_,
                            )
                        ),
                        samples,
                    )
                )
                != 0
            ):
                raise ValueError("torch incompatible inputs is incoming")

        self.buffer.extend(samples)

        if len(self.buffer) > self.max_samples:
            self.buffer = self.buffer[-self.max_samples :]

    def batch_push(self, samples: Dict) -> None:
        keys = self.keys()
        if keys:
            if set(samples.keys()) != set(keys):
                raise ValueError("keys in the inputs is not matched to the buffer data")

        keys = list(samples.keys())
        len_samples = len(samples[list(samples.keys())[0]])
        for key in keys:
            if len_samples != len(samples[key]):
                raise ValueError("Sample size is not the same among inputs keys")

        _samples = [
            dict([(key, samples[key][i]) for key in keys]) for i in range(len_samples)
        ]
        self.extends(samples=_samples)

    def sample(self, batch_size: int) -> List[Dict]:
        if not self.can_sample(batch_size=batch_size):
            raise AttributeError("Buffer doesn't have enough data to be sampled.")
        items = random.choices(self.buffer, k=batch_size)
        return items

    def torch_sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if not self.torch_compatible:
            raise RuntimeError("This method is only for torch compatible mode")
        items = self.sample(batch_size=batch_size)

        _keys = self.keys()
        if _keys is None:
            raise AssertionError

        keys = list(_keys)
        return dict(
            [
                (key, torch.stack(list(map(lambda sample: sample[key], items)), dim=0))
                for key in keys
            ]
        )
