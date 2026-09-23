# Smaller PSD bank using complete preservation trajectories

The completed small bank has 626 repair targets and 3,469 preservation
targets. Preservation comes from 246 distinct, strictly verified successful
episodes: 4–35 assistant steps per episode (median 14). All 246 assembled
episodes have exactly one final judgment step. The previous five-epoch
DP4/global-batch-32 run used 640 optimizer steps. These are attested bank
counts and one bounded read of the assembled preservation bank, not a rescan
or hash of model weights or original trajectories.

The published Qwen3.5-9B configuration records 506 repair and 815 preservation
targets (38:62), at unit target weight:
<https://github.com/essamsleiman/psd/blob/main/experiments/bfcl/configs/qwen35_9b_published.json>.
An approximately 1:1 count variant is therefore an IFV ablation, not an
exact reproduction of that configuration.

**Selection unit is the entire successful episode, never an individual step.**
Every repair target is kept. The assembled preservation bank is read once as
the authority for each episode's ordered step IDs and strict verification.
The target bank is read once to record byte offsets and must match that
authority exactly; a missing, duplicated, or out-of-order preservation step
causes failure before output. Complete episodes are grouped by episode-length
quartile. Within each stratum, a fixed seed determines a reproducible order;
whole episodes are included to approximate the requested preservation-step
budget in proportion to the source step mass of each stratum. A whole-episode
toggle can correct the overall count when a small stratum cannot fit even one
episode. The final count may differ from the request by at most the larger of
one maximum episode or 5%; it never reaches an exact count by truncation.

This choice uses no private gold, post-training result, teacher score, or
textual keyword. Selecting complete trajectories means an included episode
retains its entire action sequence and judgment. It does **not** guarantee
that the chosen subset matches the source tool-family distribution. Before
formal training, audit that distribution against trusted native-action
metadata, along with length and multimodal/context-length distributions.
If the audit reveals a material shift, freeze a revised episode-level rule
and produce a new derived bank; do not patch individual steps by hand.

On a read-only metadata trial with seed `psd-whole-episode-v2`, a requested
626-step preservation budget selects **43 complete episodes / 616 steps**;
the four length strata contribute 100, 144, 181, and 191 steps. Thus a
validated target bank would contain 626 repair + 616 preservation = 1,242
targets. Five epochs/global batch 32 would nominally require 39 steps/epoch,
or 195 optimizer steps total, instead of 640. The selected preservation
episodes contain 12,656,804 prompt token IDs and 228,739 completion token IDs,
versus 68,496,255 and 1,260,552 in the full preservation bank. These are
read-only trial figures, **not** a completed target-bank manifest or measured
wall-clock speedup; actual training time also depends on batching, image
processing, startup, and I/O.

Run `scripts/balance_psd_training_targets.py` with `--source` set to the
original combined target JSONL and `--preservation-episodes-source` set to
the corresponding assembled preservation JSONL. Write to a new output
directory. The script copies only selected original target lines by offset;
it does not hash large payloads or call any provider. Its manifest is
explicitly `prepared_for_audit_not_authorized_to_train`.

The old v190 **step-level, evenly-spaced selector must not be used** for
teacher scoring, train-ready packaging, or formal training. Neither this
episode-level prototype nor its manifest authorizes a new training run.
After selection audit, the normal exact-ID/pixel teacher top-20 and DP4 gates
must pass on a distinct derived bank; never modify the completed small-bank
checkpoint or the active old-1000 repair owner.
