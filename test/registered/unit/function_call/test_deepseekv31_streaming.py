import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepSeekV31StreamingPreamble(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    parameters={"type": "object"},
                ),
            )
        ]
        self.header = "<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>"
        self.arguments = '{"city": "Tokyo"}'
        self.call = self.header + self.arguments + "<｜tool▁call▁end｜>"

    def test_preamble_sharing_a_delta_with_a_complete_call(self):
        text = "Let me check the weather.\n<｜tool▁calls▁begin｜>" + self.call
        detector = DeepSeekV31Detector()
        result = detector.parse_streaming_increment(text, self.tools)
        expected = detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, expected.normal_text)
        self.assertEqual([call.name for call in result.calls], ["get_weather"])
        # A subsequent increment must not release the same preamble again.
        self.assertEqual(
            detector.parse_streaming_increment("", self.tools).normal_text, ""
        )

    def test_preamble_emits_before_the_call_header_arrives(self):
        detector = DeepSeekV31Detector()
        result = detector.parse_streaming_increment(
            "Checking.\n<｜tool▁calls▁begin｜>", self.tools
        )
        self.assertEqual(result.normal_text, "Checking.")
        self.assertEqual(result.calls, [])
        calls = []
        for chunk in (
            self.header,
            self.arguments[:8],
            self.arguments[8:] + "<｜tool▁call▁end｜>",
            "<｜tool▁calls▁end｜>",
        ):
            result = detector.parse_streaming_increment(chunk, self.tools)
            self.assertEqual(result.normal_text, "")
            calls.extend(result.calls)
        self.assertEqual([call.name for call in calls if call.name], ["get_weather"])
        self.assertEqual(
            json.loads("".join(call.parameters for call in calls)), {"city": "Tokyo"}
        )

    def test_preamble_before_an_unwrapped_call(self):
        detector = DeepSeekV31Detector()
        result = detector.parse_streaming_increment(
            "Checking.\n" + self.call, self.tools
        )
        self.assertEqual(result.normal_text, "Checking.")
        self.assertEqual([call.name for call in result.calls], ["get_weather"])
        self.assertEqual(
            detector.parse_streaming_increment("", self.tools).normal_text, ""
        )

    def test_call_without_preamble(self):
        result = DeepSeekV31Detector().parse_streaming_increment(
            "<｜tool▁calls▁begin｜>" + self.call, self.tools
        )
        self.assertEqual(result.normal_text, "")
        self.assertEqual([call.name for call in result.calls], ["get_weather"])

    def test_plain_text_is_unchanged(self):
        text = "  The weather is sunny.\n"
        result = DeepSeekV31Detector().parse_streaming_increment(text, self.tools)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])


if __name__ == "__main__":
    unittest.main()
