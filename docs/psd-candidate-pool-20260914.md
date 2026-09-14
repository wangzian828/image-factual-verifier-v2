# PSD candidate selection — 2026-09-14

This is a case pool for fresh, training-only Qwen rollouts. It is not a PSD
target bank, a repair-success claim, or permission to use evaluation data for
training. Current SFT and Gemini evaluation remain running independently.

## Sources and identity

- Official training metadata: `/volume/ybo/wza/data/psd-official-train-metadata-20260912`,
  8,490 cases with private references. Original pixels are in
  `jiashuhong/factcheck_train`, `factcheck_train-8490-20260907.tar.gz`.
- Pinned compressed archive: 24,127,361,006 bytes, SHA-256
  `2fda3ca7144d899e355fcbbaa4e6b93350878dbf5225ab423402123d5e37a448`.
- Actual SFT index: `/volume/ybo/wza/data/ifv-merged-4929-canonical-20260913/canonical-dataset/index.jsonl`,
  4,872 train and one validation row, plus 56 separately delivered action-only rows.
- Frozen historical split: 8,482 train and eight validation cases. Three of
  those historical validation IDs appear in the later SFT train index; all
  eight are excluded from this PSD selection. They are not independent SFT
  validation evidence. Existing training artifacts are unchanged.
- Official unified IDs reconcile one-to-one with all 4,873 reasoning-index
  IDs, including the shorter `main-*`/other delivery IDs.
- Test exclusions cover all 1,527 formal manifest IDs and all 1,526 available
  runtime image hashes. Explicit candidate/assignment aliases are also checked.

The delivery repository contains successful packages, not a complete teacher
attempt ledger. Therefore the hard-pool provenance is **not in successful
teacher delivery**, rather than an invented per-case model/API failure reason.
Missing reasoning alone does not make the 56 action-only deliveries failures.

## Initial selection

| Pool | Cases | Real | Fake |
|---|---:|---:|---:|
| Hard training candidates | 3,081 | 990 | 2,091 |
| Previously SFT-trained cases to revisit | 1,000 | 302 | 698 |
| Fixed development cases | 400 | 129 | 271 |
| Action-only reserve | 54 | 18 | 36 |

The training collection combines the first two pools: **4,081 cases**.
Development cases are held out from PSD gradients and repair-target creation.
The remaining SFT cases remain available in the source inventory; no source
trajectory or image was deleted.

Selection uses fixed seed `psd-pool-20260914-v1`. Whole groups are formed from
the historical split group, known image hashes and explicit event identities.
Development groups cannot contain any successful teacher delivery. Sampling
is proportional by label/route, and the SFT revisit pool additionally covers
teacher tool-count bins (39 with 0–5 calls, 438 with 6–12, 523 with 13+).

There are 179 quarantined cases across the complete 8,490-case inventory:
128 share a candidate/assignment identity with a test case, 44 match known
test image hashes, eight belong to existing validation, and one is related to
an excluded group. Reasons overlap. The canonical train/test case-ID
intersection itself is zero. Shared candidate IDs are conservatively
quarantined and are not by themselves proof of identical pixels or of model
memorization. No original data is removed.

## Image preparation and release gate

Server output:
`/volume/ybo/wza/data/psd-candidate-pool-20260914-v1`

The selector streams the official archive with four bounded byte-range
downloads. The compressed archive is never saved. All 8,490 original images
are hashed; only images selected for training/development are stored, with
content-addressed deduplication. Public release views use hard links.

Before release, the complete compressed SHA-256 and size must match, every
selected image must decode and retain its hash, and original-pixel overlap
with test/PSD development/historical SFT is rechecked. Newly detected overlaps
are quarantined by group and final counts are reported separately.

`selection-report.json` is authoritative. Until it says
`ready_for_fresh_policy_rollouts`, image preparation is incomplete and no
rollout should start. `selected_images_pending` is not a runnable release.

Each finalized `train`, `hard_train`, `sft_revisit`, and `development` release
has public input containing only `case_id`, `image_path`, and `image_sha256`.
Gold, labels, split roles, construction routes and teacher outcomes stay in
separate evaluator/selection artifacts. The development manifest is marked
`training_prohibited=true`.

## Execution and follow-up

Run-specific source copy:
`/volume/ybo/wza/training-artifacts/psd-candidate-pool-20260914/prepare_psd_candidate_pool.py`

The current download log is `materialize-r2.log` in the same control directory;
the earlier single-stream log remains as provenance. Source changes were
committed and pushed from the local Windows checkout, not the server. Reusing
the output requires identical source hashes, seed and requested pool sizes.
Verified images can be reused after interruption; no live provider calls or
model training are part of preparation.

After the existing SFT/evaluation sequence, freeze a chosen checkpoint and
recollect current-policy traces. Before production PSD, correct the current
preservation/source-failure gates so evidence-insufficient correct labels are
not automatically preserved or excluded from semantic repair. Then construct
verified repair and preservation distributions from the same frozen model.

Local checks: seven targeted tests cover alias ambiguity, deterministic
selection, whole-group isolation, old validation, label mismatches, ordered
parallel transport and truncated-range rejection; compileall and diff-check
pass. These checks do not replace the full archive/image/release verification.
