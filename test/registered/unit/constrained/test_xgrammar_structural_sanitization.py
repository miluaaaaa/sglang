import copy
import json
import unittest

import xgrammar as xgr

from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def wrap_format(kind, leaf):
    if kind in ("optional", "star", "plus", "repeat"):
        result = {"type": kind, "content": leaf}
        if kind == "repeat":
            result.update(min=1, max=2)
        return result
    if kind in ("dispatch", "token_dispatch"):
        return {"type": kind, "rules": [["<f>" if kind == "dispatch" else 1, leaf]]}
    if kind in ("sequence", "or"):
        return {"type": kind, "elements": [leaf]}
    tag = {"type": "tag", "begin": "<f>", "content": leaf, "end": "</f>"}
    if kind == "tag":
        return tag
    if kind == "token_triggered_tags":
        tag["begin"] = {"type": "token", "token": 1}
        return {"type": kind, "trigger_tokens": [1], "tags": [tag]}
    result = {"type": kind, "tags": [tag]}
    if kind == "triggered_tags":
        result["triggers"] = ["<f>"]
    else:
        result["separator"] = ","
    return result


CONTAINERS = (
    "tag",
    "sequence",
    "or",
    "triggered_tags",
    "tags_with_separator",
    "optional",
    "star",
    "plus",
    "repeat",
    "dispatch",
    "token_dispatch",
    "token_triggered_tags",
)


class TestStructuralFormatSanitization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vocab = ["<eos>", "<f>", "</f>"] + [chr(c) for c in range(32, 127)]
        cls.compiler = xgr.GrammarCompiler(xgr.TokenizerInfo(vocab, stop_token_ids=[0]))

    def test_all_container_types_compile_with_missing_or_null_schema(self):
        for kind in CONTAINERS:
            for missing in (False, True):
                with self.subTest(kind=kind, missing=missing):
                    leaf = {"type": "json_schema"}
                    if not missing:
                        leaf["json_schema"] = None
                    fmt = wrap_format(kind, leaf)
                    XGrammarGrammarBackend._sanitize_structural_format(fmt)
                    self.assertEqual(leaf["json_schema"], {})
                    self.compiler.compile_structural_tag(
                        json.dumps({"type": "structural_tag", "format": fmt})
                    )

    def test_nested_mixed_containers(self):
        leaf = {"type": "json_schema", "json_schema": None}
        fmt = leaf
        for kind in (
            "optional",
            "dispatch",
            "repeat",
            "or",
            "token_dispatch",
            "sequence",
        ):
            fmt = wrap_format(kind, fmt)
        XGrammarGrammarBackend._sanitize_structural_format(fmt)
        self.assertEqual(leaf["json_schema"], {})
        self.compiler.compile_structural_tag(
            json.dumps({"type": "structural_tag", "format": fmt})
        )

    def test_existing_schema_payload_is_unchanged(self):
        schema = {
            "type": "object",
            "properties": {"json_schema": {"type": "null"}},
            "default": {"type": "json_schema", "json_schema": None},
        }
        original = copy.deepcopy(schema)
        leaf = {"type": "json_schema", "json_schema": schema}
        fmt = wrap_format("optional", leaf)
        XGrammarGrammarBackend._sanitize_structural_format(fmt)
        self.assertEqual(schema, original)
        self.assertIs(leaf["json_schema"], schema)

    def test_deprecated_xml_schema_and_invalid_rules(self):
        leaf = {"type": "qwen_xml_parameter", "json_schema": None}
        fmt = wrap_format("token_dispatch", leaf)
        fmt["rules"].extend([[], [1], None])
        XGrammarGrammarBackend._sanitize_structural_format(fmt)
        self.assertEqual(leaf["json_schema"], {})
        self.assertEqual(fmt["rules"][1:], [[], [1], None])
        for value in (None, [], "not a format"):
            XGrammarGrammarBackend._sanitize_structural_format(value)


if __name__ == "__main__":
    unittest.main()
