"""Replay the same PSD batches after a native Trainer resume.

Transformers' grouped and random samplers default to the process-global RNG.
Trainer restores the checkpoint RNG *after* constructing/skipping the resumed
dataloader, so a resumed model can otherwise choose a different permutation.
The sampler needs its own seed, derived only from the dataset seed and epoch.
"""

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def checkpoint_epoch(checkpoint):
    if not checkpoint:
        return 0
    state = json.loads((Path(checkpoint) / 'trainer_state.json').read_text())
    epoch = state.get('epoch')
    if not isinstance(epoch, (int, float)) or epoch < 0:
        raise ValueError('PSD checkpoint lacks a valid epoch for sampler replay')
    return int(epoch)


def install_deterministic_psd_sampler():
    import torch
    from torch.utils.data import RandomSampler
    from transformers import Trainer
    from transformers.trainer_pt_utils import LengthGroupedSampler, get_length_grouped_indices

    if getattr(Trainer, '_ifv_psd_deterministic_sampler', False):
        return

    class EpochLengthGroupedSampler(LengthGroupedSampler):
        def __init__(self, *, lengths, batch_size, seed, epoch):
            super().__init__(lengths=lengths, batch_size=batch_size)
            self.seed = seed
            self.epoch = epoch

        def set_epoch(self, epoch):
            self.epoch = int(epoch)

        def __iter__(self):
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, generator=generator)
            digest = hashlib.sha256(','.join(map(str, indices)).encode()).hexdigest()[:16]
            logger.warning('IFV PSD sampler epoch=%d seed=%d order=%s',
                           self.epoch, self.seed, digest)
            return iter(indices)

    class EpochRandomSampler(RandomSampler):
        def __init__(self, sampler, *, seed, epoch):
            super().__init__(sampler.data_source, replacement=sampler.replacement,
                             num_samples=sampler.num_samples)
            self.seed = seed
            self.epoch = epoch

        def set_epoch(self, epoch):
            self.epoch = int(epoch)

        def __iter__(self):
            self.generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = list(super().__iter__())
            digest = hashlib.sha256(','.join(map(str, indices)).encode()).hexdigest()[:16]
            logger.warning('IFV PSD random sampler epoch=%d seed=%d order=%s',
                           self.epoch, self.seed, digest)
            return iter(indices)

    original = Trainer._get_train_sampler

    def get_train_sampler(self, train_dataset=None):
        sampler = original(self, train_dataset)
        if not isinstance(sampler, (LengthGroupedSampler, RandomSampler)):
            return sampler
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        if not isinstance(seed, int):
            raise ValueError('PSD sampler requires an integer data seed')
        epoch = checkpoint_epoch(self.args.resume_from_checkpoint)
        if isinstance(sampler, LengthGroupedSampler):
            return EpochLengthGroupedSampler(
                lengths=sampler.lengths, batch_size=sampler.batch_size, seed=seed,
                epoch=epoch)
        return EpochRandomSampler(sampler, seed=seed, epoch=epoch)

    Trainer._get_train_sampler = get_train_sampler
    Trainer._ifv_psd_deterministic_sampler = True
