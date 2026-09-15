"""Evaluate epoch 2 of the same three-epoch run with the unchanged protocol."""
import run_sft3084_eval as controller

controller.OUTPUT = controller.ROOT / 'runs/eval/qwen35-sft2056-epoch2-agent-formal1527-20260915'
controller.EXPORT = controller.ROOT / 'exports/h20-sft-merged4872-epoch2-step2056-20260915/export.json'
controller.ALIAS = 'ifv-qwen3.5-9b-sft-2056'
controller.EXPECTED_STEP = 2056
controller.EXPECTED_EPOCH = 2

if __name__ == '__main__':
    controller.main()
