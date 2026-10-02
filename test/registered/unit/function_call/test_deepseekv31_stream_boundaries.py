import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepSeekV31StreamBoundaries(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(name=name, parameters={"type": "object"}),
            )
            for name in ("get_weather", "get_time")
        ]
        self.begin = "<｜tool▁calls▁begin｜>"
        self.end = "<｜tool▁calls▁end｜>"
        self.close = "<｜tool▁call▁end｜>"

    def opener(self, name):
        return "<｜tool▁call▁begin｜>" + name + "<｜tool▁sep｜>"

    def assert_calls_match(self, chunks):
        expected = DeepSeekV31Detector().detect_and_parse("".join(chunks), self.tools)
        detector = DeepSeekV31Detector()
        accumulated = {}
        for chunk in chunks:
            for call in detector.parse_streaming_increment(chunk, self.tools).calls:
                entry = accumulated.setdefault(call.tool_index, [None, ""])
                if call.name:
                    self.assertIsNone(entry[0], "tool name emitted more than once")
                    entry[0] = call.name
                entry[1] += call.parameters
        self.assertEqual(list(accumulated), list(range(len(expected.calls))))
        self.assertEqual(
            [(name, json.loads(arguments)) for name, arguments in accumulated.values()],
            [(call.name, json.loads(call.parameters)) for call in expected.calls],
        )
        self.assertEqual(
            detector.prev_tool_call_arr,
            [
                {"name": call.name, "arguments": json.loads(call.parameters)}
                for call in expected.calls
            ],
        )

    def test_reported_shared_call_boundary(self):
        self.assert_calls_match(
            [
                self.begin + self.opener("get_weather"),
                '{"city": "Tokyo"}' + self.close + self.opener("get_weather"),
                '{"city": "Paris"}' + self.close + self.end,
            ]
        )

    def test_multiple_complete_calls_in_one_delta(self):
        self.assert_calls_match(
            [
                self.begin
                + self.opener("get_weather")
                + '{"city": "Tokyo"}'
                + self.close
                + self.opener("get_time")
                + '{"zone": "UTC"}'
                + self.close
                + self.end
            ]
        )

    def test_all_groupings_at_semantic_boundaries(self):
        pieces = [
            self.begin,
            self.opener("get_weather"),
            '{"city": "Tokyo"}',
            self.close,
            self.opener("get_time"),
            '{"zone": "UTC"}',
            self.close,
            self.end,
        ]
        for mask in range(1 << (len(pieces) - 1)):
            chunks = [pieces[0]]
            for index, piece in enumerate(pieces[1:]):
                if mask & (1 << index):
                    chunks.append(piece)
                else:
                    chunks[-1] += piece
            with self.subTest(mask=mask):
                self.assert_calls_match(chunks)

    def test_argument_deltas_then_separate_end_token(self):
        self.assert_calls_match(
            [
                self.begin + self.opener("get_weather"),
                '{"city":',
                ' "Tokyo"}',
                self.close,
                self.end,
            ]
        )


if __name__ == "__main__":
    unittest.main()
