"""DeepSeek V3.1 must consume each call even when deltas span calls."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

BEGIN = "<｜tool▁calls▁begin｜>"
END = "<｜tool▁calls▁end｜>"
CALL_BEGIN = "<｜tool▁call▁begin｜>"
CALL_END = "<｜tool▁call▁end｜>"
SEP = "<｜tool▁sep｜>"


def header(name):
    return CALL_BEGIN + name + SEP


def call(name, city):
    return header(name) + json.dumps({"city": city}) + CALL_END


class TestDeepSeekV31MultipleCalls(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(name=name, parameters={"type": "object"}),
            )
            for name in ("get_weather", "get_forecast")
        ]

    def _assert_calls(self, chunks, expected):
        detector = DeepSeekV31Detector()
        accumulated = {}
        for chunk in chunks:
            for item in detector.parse_streaming_increment(chunk, self.tools).calls:
                entry = accumulated.setdefault(
                    item.tool_index, {"name": None, "arguments": ""}
                )
                if item.name:
                    self.assertIsNone(entry["name"], "tool name was emitted twice")
                    entry["name"] = item.name
                entry["arguments"] += item.parameters or ""
        self.assertEqual(sorted(accumulated), list(range(len(expected))))
        parsed = [
            (item["name"], json.loads(item["arguments"]))
            for item in accumulated.values()
        ]
        self.assertEqual(parsed, expected)
        self.assertEqual(
            [(item["name"], item["arguments"]) for item in detector.prev_tool_call_arr],
            expected,
        )

    def test_issue_delta_spans_end_and_next_separator(self):
        self._assert_calls(
            [
                BEGIN + header("get_weather"),
                '{"city": "Tokyo"}' + CALL_END + header("get_weather"),
                '{"city": "Paris"}' + CALL_END + END,
            ],
            [("get_weather", {"city": "Tokyo"}), ("get_weather", {"city": "Paris"})],
        )

    def test_two_calls_in_one_delta(self):
        text = (
            BEGIN + call("get_weather", "Tokyo") + call("get_forecast", "Paris") + END
        )
        expected = [
            ("get_weather", {"city": "Tokyo"}),
            ("get_forecast", {"city": "Paris"}),
        ]
        self._assert_calls([text], expected)
        self.assertEqual(
            [
                (item.name, json.loads(item.parameters))
                for item in DeepSeekV31Detector()
                .detect_and_parse(text, self.tools)
                .calls
            ],
            expected,
        )

    def test_complete_call_per_delta(self):
        self._assert_calls(
            [BEGIN + call("get_weather", "Tokyo"), call("get_forecast", "Paris") + END],
            [("get_weather", {"city": "Tokyo"}), ("get_forecast", {"city": "Paris"})],
        )

    def test_complete_arguments_before_end_marker(self):
        self._assert_calls(
            [
                BEGIN + header("get_weather"),
                '{"city": "Tokyo"}',
                CALL_END + call("get_forecast", "Paris") + END,
            ],
            [("get_weather", {"city": "Tokyo"}), ("get_forecast", {"city": "Paris"})],
        )

    def test_argument_and_end_marker_character_chunks(self):
        self._assert_calls(
            [
                BEGIN + header("get_weather"),
                *list('{"city": "Tokyo"}' + CALL_END),
                call("get_forecast", "Paris") + END,
            ],
            [("get_weather", {"city": "Tokyo"}), ("get_forecast", {"city": "Paris"})],
        )

    def test_second_name_and_arguments_split(self):
        self._assert_calls(
            [
                BEGIN + call("get_weather", "Tokyo") + CALL_BEGIN + "get_",
                "forecast" + SEP + '{"city":',
                ' "Paris"}' + CALL_END + END,
            ],
            [("get_weather", {"city": "Tokyo"}), ("get_forecast", {"city": "Paris"})],
        )

    def test_single_complete_call_needs_no_followup(self):
        self._assert_calls(
            [BEGIN + call("get_weather", "Tokyo") + END],
            [("get_weather", {"city": "Tokyo"})],
        )


if __name__ == "__main__":
    unittest.main()
