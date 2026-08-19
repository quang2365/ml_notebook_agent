import unittest
from unittest.mock import patch

from model.model import create_llm


class ModelSelectionTests(unittest.TestCase):
    @patch("model.model.ChatOpenAI")
    @patch("model.model.get_api_key", return_value="test-key")
    @patch(
        "model.model.load_config",
        return_value={
            "provider": "custom",
            "model": "test-model",
            "base_url": "https://example.com/v1",
        },
    )
    def test_create_llm_from_saved_config(
        self,
        _mock_config,
        _mock_key,
        mock_chat_openai,
    ) -> None:
        create_llm()

        mock_chat_openai.assert_called_once_with(
            model="test-model",
            api_key="test-key",
            base_url="https://example.com/v1",
        )

    @patch("model.model.load_config", return_value=None)
    def test_create_llm_requires_config(self, _mock_config) -> None:
        with self.assertRaisesRegex(RuntimeError, "qiu setup"):
            create_llm()


if __name__ == "__main__":
    unittest.main()