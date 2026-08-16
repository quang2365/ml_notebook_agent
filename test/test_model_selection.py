import unittest
from unittest.mock import patch

from main import ask_use_deepseek
from model.model import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
    create_llm,
)


class ModelSelectionTests(unittest.TestCase):
    @patch("model.model.ChatOpenAI")
    def test_create_deepseek_model_when_selected(
        self,
        mock_chat_openai,
    ) -> None:
        with patch.dict(
            "os.environ",
            {"DEEPSEEK_API_KEY": "test-key"},
            clear=False,
        ):
            create_llm(use_deepseek=True)

        mock_chat_openai.assert_called_once_with(
            model=DEEPSEEK_MODEL,
            base_url=DEEPSEEK_BASE_URL,
            api_key="test-key",
            extra_body={
                "thinking": {
                    "type": "disabled",
                }
            },
        )

    @patch("model.model.ChatOpenAI")
    def test_keep_nvidia_model_when_deepseek_not_selected(
        self,
        mock_chat_openai,
    ) -> None:
        with patch.dict(
            "os.environ",
            {"NVIDIA_API_KEY": "test-key"},
            clear=False,
        ):
            create_llm(use_deepseek=False)

        mock_chat_openai.assert_called_once_with(
            model=NVIDIA_MODEL,
            base_url=NVIDIA_BASE_URL,
            api_key="test-key",
        )

    @patch("builtins.input", return_value="yes")
    def test_prompt_accepts_deepseek(self, _mock_input) -> None:
        self.assertTrue(ask_use_deepseek())

    @patch("builtins.input", return_value="")
    def test_prompt_defaults_to_current_model(self, _mock_input) -> None:
        self.assertFalse(ask_use_deepseek())


if __name__ == "__main__":
    unittest.main()
