"""Compile nested structural formats with missing schemas using real XGrammar."""

import copy
import json
import unittest

import xgrammar as xgr

from sglang.srt.constrained.base_grammar_backend import InvalidGrammarObject
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammarBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestStructuralSchemaSanitization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vocabulary = ["<eos>"] + [chr(c) for c in range(32, 127)]
        cls.tokenizer_info = xgr.TokenizerInfo(vocabulary, stop_token_ids=[0])
        cls.compiler = xgr.GrammarCompiler(cls.tokenizer_info)

    def _backend(self):
        tokenizer_info = self.tokenizer_info

        class Tokenizer:
            def init_xgrammar(self):
                return tokenizer_info, None

        backend = XGrammarGrammarBackend(Tokenizer(), vocab_size=96)
        self.addCleanup(backend.executor.shutdown)
        return backend

    def _compile(self, fmt):
        XGrammarGrammarBackend._sanitize_structural_format(fmt)
        return self.compiler.compile_structural_tag(
            {"type": "structural_tag", "format": fmt}
        )

    def _content_container(self, name, **fields):
        leaf = {"type": "json_schema", "json_schema": None}
        self._compile({"type": name, "content": leaf, **fields})
        self.assertEqual(leaf["json_schema"], {})

    def test_optional(self):
        self._content_container("optional")

    def test_star(self):
        self._content_container("star")

    def test_plus(self):
        self._content_container("plus")

    def test_repeat(self):
        self._content_container("repeat", min=1, max=2)

    def _dispatch(self, name, trigger):
        leaves = [{"type": "json_schema", "json_schema": None} for _ in range(2)]
        self._compile({"type": name, "rules": [[trigger, leaves[0]], ["x", leaves[1]]]})
        for leaf in leaves:
            self.assertEqual(leaf["json_schema"], {})

    def test_dispatch(self):
        self._dispatch("dispatch", "<f>")

    def test_token_dispatch(self):
        self._dispatch("token_dispatch", 1)

    def test_token_triggered_tags(self):
        leaf = {"type": "json_schema", "json_schema": None}
        self._compile(
            {
                "type": "token_triggered_tags",
                "trigger_tokens": [1],
                "tags": [
                    {
                        "type": "tag",
                        "begin": {"type": "token", "token": 1},
                        "content": leaf,
                        "end": "!",
                    }
                ],
            }
        )
        self.assertEqual(leaf["json_schema"], {})

    def test_nested_containers_and_missing_schema(self):
        leaf = {"type": "json_schema"}
        self._compile(
            {
                "type": "sequence",
                "elements": [
                    {
                        "type": "optional",
                        "content": {
                            "type": "dispatch",
                            "rules": [["<f>", {"type": "plus", "content": leaf}]],
                        },
                    }
                ],
            }
        )
        self.assertEqual(leaf["json_schema"], {})

    def test_existing_sequence_and_tag(self):
        leaf = {"type": "json_schema", "json_schema": None}
        self._compile(
            {
                "type": "sequence",
                "elements": [
                    {"type": "tag", "begin": "<f>", "content": leaf, "end": "</f>"}
                ],
            }
        )
        self.assertEqual(leaf["json_schema"], {})

    def test_explicit_schema_and_schema_contents_are_preserved(self):
        schema = {"type": "object", "properties": {"content": {"type": "null"}}}
        fmt = {
            "type": "optional",
            "content": {"type": "json_schema", "json_schema": schema},
        }
        original = copy.deepcopy(fmt)
        self._compile(fmt)
        self.assertEqual(fmt, original)

    def test_qwen_xml_parameter_missing_schema(self):
        leaf = {"type": "qwen_xml_parameter", "name": "arg", "json_schema": None}
        XGrammarGrammarBackend._sanitize_structural_format(
            {"type": "optional", "content": leaf}
        )
        self.assertEqual(leaf["json_schema"], {})

    def test_non_format_values_are_ignored(self):
        for value in (None, "text", [], 42):
            XGrammarGrammarBackend._sanitize_structural_format(value)

    def test_boolean_schemas_are_preserved(self):
        for schema in (True, False):
            fmt = {
                "type": "optional",
                "content": {"type": "json_schema", "json_schema": schema},
            }
            XGrammarGrammarBackend._sanitize_structural_format(fmt)
            self.assertIs(fmt["content"]["json_schema"], schema)

    def test_malformed_dispatch_rules_remain_validation_errors(self):
        for kind in ("dispatch", "token_dispatch"):
            invalid_rules = [
                [rule] for rule in ([], ["x"], ["x", {}, {}], None, "x")
            ] + [None, 42, {}, "x"]
            for rules in invalid_rules:
                with self.subTest(kind=kind, rules=rules):
                    fmt = {"type": kind, "rules": rules}
                    original = copy.deepcopy(fmt)
                    XGrammarGrammarBackend._sanitize_structural_format(fmt)
                    self.assertEqual(fmt, original)
                    with self.assertRaises(RuntimeError):
                        self.compiler.compile_structural_tag(
                            {"type": "structural_tag", "format": fmt}
                        )

    def test_backend_dispatch_accepts_all_nested_containers(self):
        backend = self._backend()
        leaf = {"type": "json_schema", "json_schema": None}
        formats = [
            {"type": kind, "content": leaf} for kind in ("optional", "star", "plus")
        ] + [
            {"type": "repeat", "min": 1, "max": 2, "content": leaf},
            {"type": "dispatch", "rules": [["<f>", leaf]]},
            {"type": "token_dispatch", "rules": [[1, leaf]]},
            {
                "type": "token_triggered_tags",
                "trigger_tokens": [1],
                "tags": [
                    {
                        "type": "tag",
                        "begin": {"type": "token", "token": 1},
                        "content": leaf,
                        "end": "!",
                    }
                ],
            },
        ]
        for fmt in formats:
            with self.subTest(kind=fmt["type"]):
                result = backend.dispatch_structural_tag(
                    json.dumps({"type": "structural_tag", "format": fmt})
                )
                self.assertNotIsInstance(result, InvalidGrammarObject)

    def test_backend_dispatch_returns_invalid_object_for_malformed_rules(self):
        backend = self._backend()
        for kind in ("dispatch", "token_dispatch"):
            for rules in (None, [["x"]], [["x", {}, {}]]):
                with self.subTest(kind=kind, rules=rules):
                    result = backend.dispatch_structural_tag(
                        json.dumps(
                            {
                                "type": "structural_tag",
                                "format": {"type": kind, "rules": rules},
                            }
                        )
                    )
                    self.assertIsInstance(result, InvalidGrammarObject)


if __name__ == "__main__":
    unittest.main()
