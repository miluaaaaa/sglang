import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepSeekV31StreamPreamble(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(name="get_weather", parameters={"type": "object"}),
            )
        ]
        self.opener = "<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>"
        self.arguments = '{"city": "Tokyo"}'
        self.closer = "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"

    def test_complete_call_preserves_preamble(self):
        detector = DeepSeekV31Detector()
        text = (
            "Let me check."
            + detector.bot_token
            + self.opener
            + self.arguments
            + self.closer
        )
        expected = detector.detect_and_parse(text, self.tools)
        result = detector.parse_streaming_increment(text, self.tools)
        self.assertEqual(result.normal_text, expected.normal_text)
        self.assertEqual(result.calls[0].name, "get_weather")

    def test_preamble_is_not_repeated_with_argument_deltas(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped):
                detector = DeepSeekV31Detector()
                prefix = detector.bot_token if wrapped else ""
                results = [
                    detector.parse_streaming_increment(chunk, self.tools)
                    for chunk in (
                        "Let me check.\n" + prefix + self.opener,
                        self.arguments,
                        self.closer,
                    )
                ]
                self.assertEqual(
                    [result.normal_text for result in results],
                    ["Let me check.\n", "", ""],
                )
                calls = [call for result in results for call in result.calls]
                self.assertEqual(
                    [call.name for call in calls if call.name], ["get_weather"]
                )
                self.assertEqual(
                    json.loads("".join(call.parameters for call in calls)),
                    {"city": "Tokyo"},
                )

    def test_prefix_before_section_without_function_header(self):
        detector = DeepSeekV31Detector()
        first = detector.parse_streaming_increment(
            "Checking." + detector.bot_token, self.tools
        )
        second = detector.parse_streaming_increment(self.opener, self.tools)
        self.assertEqual(first.normal_text, "Checking.")
        self.assertEqual(first.calls, [])
        self.assertEqual(second.normal_text, "")
        self.assertEqual(second.calls[0].name, "get_weather")

    def test_separately_streamed_text_and_no_preamble(self):
        detector = DeepSeekV31Detector()
        first = detector.parse_streaming_increment("Checking.", self.tools)
        second = detector.parse_streaming_increment(
            detector.bot_token + self.opener, self.tools
        )
        self.assertEqual(first.normal_text, "Checking.")
        self.assertEqual(second.normal_text, "")
        self.assertEqual(second.calls[0].name, "get_weather")


if __name__ == "__main__":
    unittest.main()
