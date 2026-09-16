"""Opt-in vLLM 0.18.1 parser plugin; no package patch or Agent change."""
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

from scripts.server.psd_single_tool_schema import single_call_schema


@ToolParserManager.register_module('ifv_psd_qwen3_single', force=False)
class PSDSingleToolParser(Qwen3CoderToolParser):
    def adjust_request(self, request):
        request = super().adjust_request(request)
        if (isinstance(request, ChatCompletionRequest) and request.tool_choice == 'required'
                and request.parallel_tool_calls is False):
            if request.structured_outputs is None or request.structured_outputs.json is None:
                raise ValueError('Required tool request has no generated constraint')
            request.structured_outputs.json = single_call_schema(request.structured_outputs.json)
        return request
