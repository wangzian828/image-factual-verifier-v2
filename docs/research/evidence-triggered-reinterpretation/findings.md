# Research Findings

## Research Question

Can the existing Agent use newly retrieved knowledge to discover a fine-grained visual distinction, return to the original image, and causally revise a later investigation Decision? If not, where does that chain first break?

## Current Understanding

The formal v4 runtime already contains a bounded Evidence-to-Vision control path, but structural availability is not behavioral evidence. The current study therefore treats the unmodified runtime as the baseline and diagnoses the earliest broken stage on deliberately selected cases.

## Key Results

No diagnostic case has been run yet. The two-GPU student endpoint passed its serving gates.

## Lessons and Constraints

- A `before / after / cause` narrative that does not change a subsequent action or Decision is not a successful mechanism.
- Web Evidence need not trigger reinspection when it already resolves the relevant question; reinspection is useful only when it creates a new image-checkable distinction.
- Evaluator-private annotations may support offline candidate selection and post-run audit, but may not enter the runtime or guide its queries.
- The first intervention, if any, must target the earliest recurring failure stage.

## Open Questions

- Does v4 retrieve the discriminative knowledge on suitable cases?
- Does Discrepancy Decision recognize when that knowledge makes the original image worth rechecking?
- Can the focused inspection resolve genuinely fine-grained details at available image resolution?
- If it can, does the following Decision actually consume the new observation?

## Experiment Trajectory

Baseline protocol locked; zero case runs completed.

