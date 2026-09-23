# Count-balanced PSD training variant

The completed small bank has 626 repair and 3,469 preservation targets. The
preservation targets come from 246 unique verified cases, averaging 14.1
assistant steps per case. Its five-epoch DP4/global-batch-32 run used 640
optimizer steps. These figures come from the attested small-bank manifests, not
from a rescan of model or raw trajectory payloads.

The published Qwen3.5-9B configuration records 506 repair and 815 preservation
targets (38:62), each at unit weight, with no aggregate rebalancing:
<https://github.com/essamsleiman/psd/blob/main/experiments/bfcl/configs/qwen35_9b_published.json>.
Thus a 1:1 **target-count** variant is deliberately more repair-heavy than
the published configuration; it must not replace the original baseline's
identity or be described as exact upstream parity.

For a faster, explicitly count-balanced IFV variant, keep every verified
repair and select an equal number of preservation targets. Spread selections
across verified preservation cases before taking a second or third step from
any one case; within a case choose evenly spaced steps. Do not select by
gold labels, post-training outcomes, or teacher score. Retain all source data
unchanged and write a new scoped target bank. Use the selector **before**
frozen-teacher top-20 scoring so discarded steps consume neither teacher GPU
time nor optimizer time. Existing scored banks can also be subsetted without
re-scoring the retained exact targets.

On the completed small bank this would select 626 repair + 626 preservation =
1,252 targets. At the same batch size and five epochs, the nominal schedule is
40 steps/epoch and 200 optimizer steps total, versus 128 and 640 previously.
That is a 68.75% step reduction, not a measured wall-clock speedup; prompt
lengths, batching, startup, and I/O still matter. The original target weights
remain 1.0. If closer-to-published count mix is desired instead, cap
preservation near `round(626 * 815 / 506) = 1,008`, giving 1,634 targets and
about 260 five-epoch steps.

The selector is `scripts/balance_psd_training_targets.py`. It streams metadata
and records byte offsets, then seeks only selected original lines; it neither
copies all preservation rows nor hashes large payloads. It emits a small
manifest with source identity, counts, per-case quotas, and the new target
file. This is a preparation utility, not authorization to start formal
training. Before a new run, bind the selected target file through the normal
teacher-scoring/datum and DP4 gates, verify final counts and exact media/ID
provenance, and use a distinct experiment name. Never modify the completed
small-bank checkpoint or silently change an active old-1000 repair owner.
