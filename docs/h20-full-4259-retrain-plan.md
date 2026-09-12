# H20 merged canonical SFT retrain protocol — 2026-09-12

## Decision

The partial canonical-v2 run was intentionally interrupted at optimizer step
259/1,004 after a second raw package became available. It produced no accepted
checkpoint and is not an effectiveness result. The replacement experiment must
start from the clean Qwen3.5-9B base and train once over the merged canonical
source, not resume the interrupted weights.

## Frozen source inventory

- Initial canonical package: 2,607 raw traces, archive SHA-256
  `6460fe4f55f5ace2eb4ad43d811531fa1d5405a663ff8f0fa59e4b8823ff18d2`.
- Quality-reroll package 02: 1,652 raw traces, archive SHA-256
  `153dac2e15efcc46385f35af158213f6d5720363a6475641ad4ed54f18729119`.
- Case-ID overlap: 0. Episode-ID overlap: 0. Combined raw inventory: 4,259.
- Package 02 canonical trace audit: 1,652/1,652 passed, with 759 retained
  protocol-correction warnings and no scheduler/protocol/route rejection
  failures in the source audit.

These are raw inventory counts. The trainable complete-reasoning count is fixed
only after canonical conversion; action-only traces remain archived and are not
silently converted into reasoning supervision.

## Blocking media gate

Package 02 intentionally omits external media and runtime stores. On the H20
host, the existing initial delivery resolves only 7/1,652 initial image hashes.
The other 1,645 initial images are unavailable. In addition, 1,373 traces expose
more than one image to the policy; their archived requests contain at least
6,191 image slots beyond the initial image. None of the package-02 runtime-store
paths is accessible on H20.

Do not export package 02 as text-only or initial-image-only data. Before merged
training, obtain an attested package-02 media projection containing either:

1. the exact runtime-store context manifests and content-addressed artifacts,
   plus the initial images; or
2. a frozen policy-media sidecar with ordered image hashes, message marker
   positions, image files and checksums for every episode.

## Mandatory release gates

1. Verify both archive hashes and every indexed trace hash.
2. Preserve every raw row; report reasoning and action-only counts separately.
3. Reject duplicate case IDs, episode IDs, trace hashes and overlap with the
   frozen 1,527-case evaluation set.
4. Rebuild targets only from canonical traces. Media sidecars may supply bytes,
   order and marker positions, never target text.
5. Require every final observation ID to be visible in a preceding successful,
   positively supervised observation.
6. Normalize recoverable executed tool arguments and mask only irreparable or
   runtime-rejected action/thought targets; never drop the containing episode.
7. Run the independent rebuild audit and the real Qwen processor over every row
   with max length 131,072, packing length 65,536, SP4 and truncation-by-error.
8. Bind the final dataset, processor report, independent audit and policy audit
   hashes into a new training plan and a new output directory.

## Training and checkpoint policy

- Clean Qwen3.5-9B base; full-parameter BF16; four H20 GPUs; FSDP2 + SP4.
- One epoch, 131,072 model ceiling and 65,536 packing length.
- Save model-only checkpoints every 400 optimizer steps and retain the latest
  three. Optimizer, scheduler and RNG state are intentionally not saved.
- Model-only checkpoints can seed a new training stage but cannot exactly resume
  optimizer/scheduler/RNG state. This is an explicit storage tradeoff requested
  by the user.
- Stop the owned GPU keepers only after every launch gate passes; restore them on
  every exit path.

After training, verify checkpoint loadability and parameter coverage, then run
the frozen full Agent evaluation and Gemini 3.7 Flash/low judge using the fixed
1,527 denominator.
