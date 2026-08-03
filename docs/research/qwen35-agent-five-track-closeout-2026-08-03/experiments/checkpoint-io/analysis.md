# H4 analysis: checkpoint I/O

Status: confirmed safe instrumentation and bounded optimization.

Historical full resumable checkpoints are 122.71 GiB. Phase profiling measured:

| World size | Total write window | Model/assets | Optimizer |
|---:|---:|---:|---:|
| 2 | 111.03 s | 17.546 GiB / 25.86 s | 105.163 GiB / 83.85 s |
| 4 | 31.18 s | 17.546 GiB / 10.23 s | 105.163 GiB / 20.51 s |

Implemented gates cover storage capacity/filesystem preflight, post-save
checkpoint integrity, model/assets, DeepSpeed model state, optimizer state, and
recovery-finalize timing.

Rejected shortcuts:

- removing optimizer/scheduler/RNG state would invalidate exact resume;
- DLRover Flash Checkpoint is not installed and would stage into `/dev/shm`,
  which is unsafe against historical process-tree RSS on this host;
- node-local storage does not redirect this single-node GPFS save;
- unmounted local RAID0 is outside project authority.

Four-way sharding plus `save_total_limit=1` is the safe production choice.

Artifact:

- `/gsdata/home/wza/image-factual-verifier-v2-data/training/logs/checkpoint-io-13bb645-20260803/`

