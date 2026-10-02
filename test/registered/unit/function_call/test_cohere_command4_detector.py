import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCohereCommand4Streaming(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                ),
            )
        ]
        self.body = json.dumps(
            [
                {
                    "tool_call_id": "0",
                    "tool_name": "get_weather",
                    "parameters": {"city": "Paris"},
                }
            ]
        )
        self.block = "<|START_ACTION|>" + self.body + "<|END_ACTION|>"

    def _parser(self):
        return FunctionCallParser(self.tools, "cohere_command4")

    def test_preamble_and_complete_action_emit_together(self):
        parser = self._parser()
        text = "Let me check.\n" + self.block
        result = parser.parse_stream_chunk(text)
        self.assertEqual(result, self._parser().parse_non_stream(text))
        self.assertEqual(len(result[1]), 1)
        self.assertEqual(parser.detector._buffer, "")
        self.assertEqual(parser.parse_stream_end(), ("", []))

    def test_chunk_boundaries_match_non_streaming(self):
        text = "Let me check.\n" + self.block
        expected = self._parser().parse_non_stream(text)
        for split in range(1, len(text)):
            with self.subTest(split=split):
                parser = self._parser()
                normal, calls = "", []
                for chunk in (text[:split], text[split:]):
                    prefix, delta = parser.parse_stream_chunk(chunk)
                    normal += prefix
                    calls.extend(delta)
                self.assertEqual((normal, calls), expected)
                self.assertEqual(parser.detector._buffer, "")

    def test_preamble_emits_before_incomplete_action(self):
        parser = self._parser()
        self.assertEqual(
            parser.parse_stream_chunk("Checking.\n<|START_ACTION|>" + self.body),
            ("Checking.\n", []),
        )
        normal, calls = parser.parse_stream_chunk("<|END_ACTION|>")
        self.assertEqual(normal, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(json.loads(calls[0].parameters), {"city": "Paris"})

    def test_complete_action_without_preamble(self):
        self.assertEqual(
            self._parser().parse_stream_chunk(self.block),
            self._parser().parse_non_stream(self.block),
        )

    def test_plain_text_is_unchanged(self):
        text = "The weather in Paris is sunny."
        parser = self._parser()
        self.assertEqual(parser.parse_stream_chunk(text), (text, []))
        self.assertEqual(parser.detector._buffer, "")

    def test_malformed_action_preserves_preamble(self):
        text = "Checking.\n<|START_ACTION|>not JSON<|END_ACTION|>"
        parser = self._parser()
        self.assertEqual(parser.parse_stream_chunk(text), ("Checking.\n", []))
        self.assertEqual(parser.detector._buffer, "")


if __name__ == "__main__":
    unittest.main()
