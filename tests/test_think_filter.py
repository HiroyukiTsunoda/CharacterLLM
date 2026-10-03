"""
ChatEngine._filter_think_stream のキャラクタライゼーションテスト。

ストリーミング思考フィルタの分割リファクタリング前後で
挙動が変わらないことを担保する。

実行: venv\\Scripts\\python.exe -m unittest tests.test_think_filter -v
"""

import sys
import unittest
from unittest.mock import MagicMock

# llama_cpp / openai の実体は不要（GPU初期化を避ける）。import前にモック注入する。
for _mod in ("llama_cpp",):
    if _mod not in sys.modules:
        _mock = MagicMock()
        _mock.Llama = MagicMock
        sys.modules[_mod] = _mock

from src.core.chat_engine import ChatEngine  # noqa: E402

START = ChatEngine.THINK_STREAM_START
END = ChatEngine.THINK_STREAM_END


def make_engine(template_inserts_think: bool = False) -> ChatEngine:
    """モックプロバイダーで ChatEngine を構築する。"""
    loader = MagicMock()
    loader.template_inserts_think = template_inserts_think
    provider = MagicMock()
    provider.template_inserts_think = template_inserts_think
    return ChatEngine(
        model_loader=loader,
        character_manager=MagicMock(),
        openai_provider=provider,
    )


def run_filter(
    tokens: list[str], template_inserts_think: bool = False
) -> tuple[str, str, ChatEngine]:
    """フィルタを実行し (応答テキスト, 思考テキスト, engine) を返す。"""
    engine = make_engine(template_inserts_think)
    full = "".join(engine._filter_think_stream(iter(tokens)))

    think_parts: list[str] = []
    response = full
    while START in response:
        pre, rest = response.split(START, 1)
        if END in rest:
            think, tail = rest.split(END, 1)
        else:
            think, tail = rest, ""
        think_parts.append(think)
        response = pre + tail
    return response, "".join(think_parts), engine


class TestThinkFilter(unittest.TestCase):
    def test_plain_short_text(self):
        """思考なしの短文はそのまま応答になる。"""
        response, think, _ = run_filter(["こん", "にちは", "です"])
        self.assertEqual(response, "こんにちはです")
        self.assertEqual(think, "")

    def test_plain_long_text(self):
        """思考なしの長文はそのまま応答になる。"""
        tokens = ["これは長い応答テキストです。", "マーカーは含まれません。"]
        response, think, _ = run_filter(tokens)
        self.assertEqual(response, "".join(tokens))
        self.assertEqual(think, "")

    def test_think_tags(self):
        """<think>...</think> が思考として分離される。"""
        response, think, engine = run_filter(
            ["<think>", "考え中", "</think>", "応答", "です"]
        )
        self.assertEqual(response, "応答です")
        self.assertEqual(think, "考え中")
        self.assertEqual(engine.get_last_think_content(), "考え中")

    def test_template_inserted_close_think(self):
        """テンプレート自動挿入パターン（</think>開始）。"""
        response, think, _ = run_filter(
            ["</think>", "思考", "</think>", "了解した"],
            template_inserts_think=True,
        )
        self.assertEqual(response, "了解した")
        self.assertEqual(think, "思考")

    def test_implicit_think(self):
        """タグなし開始＋template_inserts_think=True は暗黙の思考になる。"""
        response, think, _ = run_filter(
            ["これは推論の", "途中テキストです", "</think>", "本文応答"],
            template_inserts_think=True,
        )
        self.assertEqual(response, "本文応答")
        self.assertEqual(think, "これは推論の途中テキストです")

    def test_gpt_oss_analysis_channel(self):
        """gpt-oss analysis チャネルが思考として分離される。"""
        response, think, engine = run_filter(
            [
                "<|channel|>",
                "analysis<|message|>分析中<|end|>",
                "<|channel|>final<|message|>",
                "最終応答",
            ]
        )
        self.assertEqual(response, "最終応答")
        self.assertEqual(think, "分析中")
        self.assertEqual(engine.get_last_think_content(), "分析中")

    def test_section_markers(self):
        """思考内容/応答内容 セクションマーカーが分離される。"""
        response, think, _ = run_filter(
            ["思考内容\n", "ここで考える\n", "応答内容\n", "やあ！"]
        )
        self.assertEqual(response, "やあ！")
        self.assertEqual(think.strip(), "ここで考える")

    def test_think_after_passthrough(self):
        """応答の途中で <think> が出現しても分離される。"""
        response, think, _ = run_filter(
            [
                "これは長い応答テキストです。",
                "<think>",
                "後から考える",
                "</think>",
                "続きの応答",
            ]
        )
        self.assertEqual(response, "これは長い応答テキストです。続きの応答")
        self.assertEqual(think, "後から考える")

    def test_end_marker_truncates(self):
        """<|end|> 以降の繰り返しテキストは出力されない。"""
        response, think, _ = run_filter(
            ["応答テキストはこちらです。", "<|end|>無視されるべき繰り返し"]
        )
        self.assertEqual(response, "応答テキストはこちらです。")
        self.assertEqual(think, "")

    def test_unterminated_think_at_stream_end(self):
        """閉じタグなしで終わった思考はストリーム終端で flush される。"""
        response, think, _ = run_filter(["<think>", "未完の思考"])
        self.assertEqual(response, "")
        self.assertEqual(think, "未完の思考")


if __name__ == "__main__":
    unittest.main()
