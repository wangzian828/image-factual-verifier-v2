import json
import random
import sys
from types import ModuleType, SimpleNamespace

import pytest

from ifv_training.psd_deterministic_sampler import (
    checkpoint_epoch, install_deterministic_psd_sampler,
)


def test_checkpoint_epoch_rejects_missing_or_invalid_state(tmp_path):
    assert checkpoint_epoch(None) == 0
    with pytest.raises(FileNotFoundError):
        checkpoint_epoch(tmp_path)
    (tmp_path / 'trainer_state.json').write_text(json.dumps({'epoch': -1}))
    with pytest.raises(ValueError):
        checkpoint_epoch(tmp_path)
    (tmp_path / 'trainer_state.json').write_text(json.dumps({'epoch': 2.75}))
    assert checkpoint_epoch(tmp_path) == 2


def test_grouped_batches_replay_independently_of_process_rng(monkeypatch, tmp_path):
    class Generator:
        def manual_seed(self, seed):
            self.seed = seed
            return self

    class LengthGroupedSampler:
        def __init__(self, *, lengths, batch_size):
            self.lengths, self.batch_size = lengths, batch_size

    class RandomSampler:
        def __init__(self, data_source, *, replacement=False, num_samples=None):
            self.data_source = data_source
            self.replacement = replacement
            self.num_samples = num_samples or len(data_source)
            self.generator = None

        def __iter__(self):
            indices = list(range(len(self.data_source)))
            random.Random(self.generator.seed).shuffle(indices)
            return iter(indices)

    class Trainer:
        def _get_train_sampler(self, train_dataset=None):
            if self.args.train_sampling_strategy == 'random':
                return RandomSampler(self.train_dataset)
            return LengthGroupedSampler(lengths=list(range(20)), batch_size=4)

    def grouped(lengths, batch_size, generator):
        indices = list(range(len(lengths)))
        random.Random(generator.seed).shuffle(indices)
        return indices

    torch = ModuleType('torch')
    torch.Generator = Generator
    torch.utils = ModuleType('torch.utils')
    torch.utils.data = ModuleType('torch.utils.data')
    torch.utils.data.RandomSampler = RandomSampler
    transformers = ModuleType('transformers')
    transformers.Trainer = Trainer
    pt_utils = ModuleType('transformers.trainer_pt_utils')
    pt_utils.LengthGroupedSampler = LengthGroupedSampler
    pt_utils.get_length_grouped_indices = grouped
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'torch.utils', torch.utils)
    monkeypatch.setitem(sys.modules, 'torch.utils.data', torch.utils.data)
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    monkeypatch.setitem(sys.modules, 'transformers.trainer_pt_utils', pt_utils)
    install_deterministic_psd_sampler()
    install_deterministic_psd_sampler()  # patch must be idempotent

    args = SimpleNamespace(data_seed=42, seed=0, resume_from_checkpoint=None,
                           train_sampling_strategy='group_by_length')
    trainer = Trainer()
    trainer.args = args
    trainer.train_dataset = list(range(20))
    baseline = trainer._get_train_sampler()
    first_epoch = list(baseline)
    random.seed(999)
    assert list(baseline) == first_epoch
    baseline.set_epoch(1)
    assert list(baseline) != first_epoch

    (tmp_path / 'trainer_state.json').write_text(json.dumps({'epoch': 0.01}))
    args.resume_from_checkpoint = str(tmp_path)
    resumed = trainer._get_train_sampler()
    random.seed(12345)
    assert list(resumed)[4:] == first_epoch[4:]

    args.train_sampling_strategy = 'random'
    args.resume_from_checkpoint = None
    random_baseline = trainer._get_train_sampler()
    random_first_epoch = list(random_baseline)
    random.seed(444)
    assert list(random_baseline) == random_first_epoch
    random_baseline.set_epoch(1)
    assert list(random_baseline) != random_first_epoch
    args.resume_from_checkpoint = str(tmp_path)
    random_resumed = trainer._get_train_sampler()
    random.seed(555)
    assert list(random_resumed)[4:] == random_first_epoch[4:]
