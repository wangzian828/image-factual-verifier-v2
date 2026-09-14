# PSD candidate selection — 2026-09-14

## Current: v2, no training-data holdout

The user chose to observe PSD development on a fixed subset of the existing
formal test set, without putting its cases/labels into gradient training or
repair targets. The complete 1,527-case reporting denominator is unchanged.
The subset list is not created by this data-preparation step. If its scores
are used for checkpoint/parameter selection, that evaluation use should be
disclosed; it is not gradient training.

The previously reserved 400 training-source cases have all been returned to
the hard training pool. The accepted **v2** release contains:

| Pool | Cases | Real | Fake |
|---|---:|---:|---:|
| Hard training candidates | 3,481 | 1,119 | 2,362 |
| Previously SFT-trained cases to revisit | 1,000 | 302 | 698 |
| Combined training candidates | **4,481** | **1,421** | **3,060** |
| Training-source development holdout | **0** | 0 | 0 |

Current collection input:
`/volume/ybo/wza/data/psd-candidate-pool-20260914-v2/train/runtime-release/runtime_input/cases.jsonl`

Private references and train allowlist are under the v2 `train/evaluator_private/`.
An exact set check verified v2 train = v1 train union v1 development, and the
1,000-case SFT revisit membership is unchanged. v2 uses the same 4,463 verified
original images via hardlinks: no redownload or large image copy. The original
v1 release remains immutable, but it is superseded for future PSD collection.
The empty v2 `development` release is structural only and is not an eval input.

The selector now defaults to `--dev-size 0`; `--reuse-verified-images-from`
binds reuse to accepted manifest/inventory hashes. v2 was generated using
`prepare_psd_candidate_pool_v2.py`, with `materialize-v2.log`, under
`/volume/ybo/wza/training-artifacts/psd-candidate-pool-20260914`.
31 targeted/regression tests pass. PSD training has not started.

## Original v1 preparation details (superseded split only)

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
- Frozen historical split: 8,482 train and eight validation cases; all eight
  validation cases are excluded from this PSD selection. Existing training
  artifacts are unchanged.
- Official unified IDs reconcile one-to-one with all 4,873 reasoning-index
  IDs, including the shorter `main-*`/other delivery IDs.
- Test exclusions cover all 1,527 formal manifest IDs and all 1,526 available
  runtime image hashes. Explicit candidate/assignment aliases are also checked.

The delivery repository contains successful packages, not a complete teacher
attempt ledger. Therefore the hard-pool provenance is **not in successful
teacher delivery**, rather than an invented per-case model/API failure reason.
Missing reasoning alone does not make the 56 action-only deliveries failures.

## Final candidate selection

| Pool | Cases | Real | Fake |
|---|---:|---:|---:|
| Hard training candidates | 3,081 | 990 | 2,091 |
| Previously SFT-trained cases to revisit | 1,000 | 302 | 698 |
| Fixed development cases | 400 | 129 | 271 |
| Action-only reserve | 54 | 18 | 36 |

The training collection combines the first two pools: **4,081 cases**.
Training composition is 1,292 real / 2,789 fake; hard candidates contribute
75.50% and SFT revisit cases 24.50%. These are input cases, not yet successful
PSD repair/preservation targets. The 54 action-only reserve cases are not
silently mixed into the collection.
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
are hashed, including a bounded 1024/JPEG95 fingerprint using the runtime's
encoder. Only images selected for training/development are stored, with
content-addressed deduplication. Public release views use hard links.
Backward tar hardlinks are resolved without interpreting filesystem links;
a target-only second pass is allowed if a selected link references an
unselected original. Other PIL-decodable formats retain their original bytes;
the runtime still sends bounded JPEG, as before. Unsafe/forward links fail
closed. Image writes use a temporary file followed by atomic replacement.

Before release, the complete compressed SHA-256 and size must match, every
selected image must decode and retain its hash, and raw/normalized-image
overlap with test/PSD development/historical SFT is rechecked. Image-only
inputs with conflicting private labels are also quarantined. Newly detected
overlaps are quarantined by group and final counts are reported separately.

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
`/volume/ybo/wza/training-artifacts/psd-candidate-pool-20260914/prepare_psd_candidate_pool_final.py`

The current download log is `materialize-r5.log` in the same control directory;
the earlier logs remain as preparation diagnostics. Source changes were
committed and pushed from the local Windows checkout, not the server. Reusing
the output requires identical source hashes, seed and requested pool sizes.
Verified images can be reused after interruption; no live provider calls or
model training are part of preparation.

After the existing SFT/evaluation sequence, freeze a chosen checkpoint and
recollect current-policy traces. Before production PSD, correct the current
preservation/source-failure gates so evidence-insufficient correct labels are
not automatically preserved or excluded from semantic repair. Then construct
verified repair and preservation distributions from the same frozen model.

`scripts/prepare_psd_image_fingerprints.py` creates the bound formal-test
fingerprint input accepted by `--test-image-audit`. It does not evaluate a
model. `scripts/verify_psd_candidate_pool.py` independently checks finalized
source/public-image hashes, public/private membership, split guards, and
case/group/raw/normalized-image isolation between train and development.
Its successful report is `release-verification.json`.

The server preparation completed successfully at 22:24 CST on 2026-09-14:

- The full 24,127,361,006-byte compressed source matched its pinned SHA-256.
- All 8,490 official image members were fingerprinted, including one tar
  hardlink. Final image checks required no additional candidate removals.
- Selected inputs contain 4,463 distinct originals: 4,227 JPEG, 140 PNG,
  92 WebP, two AVIF and two MPO. Content-deduplicated original bytes total
  12,785,588,976 (11.91 GiB); the whole prepared directory reports about 13G.
- All four public releases and their private-gold/split memberships passed
  independent verification. All 4,463 unique public images were rehashed.
- The subsequent `--wire-images` acceptance also passed for all 4,463 images
  through the actual `controlled_image_to_data_url` runtime entry point:
  bounded JPEG output, original SHA-256 and normalized SHA-256 all matched.
  AVIF and MPO originals are therefore covered by the real serialization
  path, not merely a file-extension check. No external model call was made.
- The finalized report status is `ready_for_fresh_policy_rollouts`.
  No PSD model training, repair search or provider rollout was started.

Combined collection input:
`/volume/ybo/wza/data/psd-candidate-pool-20260914-v1/train/runtime-release/runtime_input/cases.jsonl`

Its sibling `train/evaluator_private/case_split.jsonl` and
`train/evaluator_private/private_gold.jsonl` remain evaluator-only. Source
policy is `train/runtime-release/evaluator_private/source_access_policy.json`.
The corresponding fixed held-out input is
`development/runtime-release/runtime_input/cases.jsonl` under the pool root.

Local checks: 15 targeted tests cover alias ambiguity, deterministic
selection, whole-group isolation, old validation, label mismatches, ordered
parallel transport, truncated ranges, safe/unsafe hardlinks, JPEG/BMP/TIFF,
conflicting image labels, and public-release/development guards; compileall
and diff-check pass. Together with release-adapter/case-scheduler regression
checks, 29 tests pass. These checks do not replace full archive verification.
