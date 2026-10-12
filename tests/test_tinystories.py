from pathlib import Path
import tempfile
import unittest

from gator.data import (
    iter_tinystories,
    split_story_sentences,
    tinystories_sentence_pairs,
)
from gator_language_train import _fractional_source_weight


STORY = (
    "One day, Mia found a red ball. She took it to the park. "
    'Tom asked, "Can I play?" Mia smiled and said yes. '
    "They kicked the ball together. Then they went home."
)


class TinyStoriesTests(unittest.TestCase):
    def test_fractional_weight_preserves_existing_relative_mix(self) -> None:
        existing = [1.0, 4.0, 3.0, 2.0]
        tiny = _fractional_source_weight(existing, 0.40)
        weights = [tiny, *existing]
        total = sum(weights)

        self.assertAlmostEqual(weights[0] / total, 0.40)
        self.assertEqual(
            [round(weight / total, 2) for weight in weights],
            [0.40, 0.06, 0.24, 0.18, 0.12],
        )

    def test_sentence_split_retains_dialogue_punctuation(self) -> None:
        sentences = split_story_sentences(STORY)

        self.assertEqual(len(sentences), 6)
        self.assertEqual(sentences[0], "One day, Mia found a red ball.")
        self.assertEqual(sentences[2], 'Tom asked, "Can I play?"')

    def test_official_text_delimiter_preserves_multiline_stories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "TinyStories-train.txt"
            path.write_text(
                "First sentence.\nSecond sentence.\n"
                "<|endoftext|>\n"
                "Another story starts. It also ends.\n"
                "<|endoftext|>\n",
                encoding="utf-8",
            )

            stories = list(iter_tinystories(path))

        self.assertEqual(
            stories,
            [
                (0, "First sentence.\nSecond sentence."),
                (1, "Another story starts. It also ends."),
            ],
        )

    def test_context_growth_adds_prefix_sentence_pairs(self) -> None:
        at_64 = list(
            tinystories_sentence_pairs(
                [(7, STORY)],
                context_tokens=64,
                response_tokens=64,
            )
        )
        at_128 = list(
            tinystories_sentence_pairs(
                [(7, STORY)],
                context_tokens=128,
                response_tokens=64,
            )
        )

        self.assertEqual(len(at_64), 1)
        self.assertEqual(len(at_128), 2)
        self.assertIn("She took it to the park.", at_64[0].context[0].text)
        self.assertEqual(
            at_64[0].response,
            'Tom asked, "Can I play?" Mia smiled and said yes.',
        )
        self.assertIn("Mia smiled and said yes.", at_128[1].context[0].text)
        self.assertEqual(
            at_128[1].response,
            "They kicked the ball together. Then they went home.",
        )

    def test_response_growth_adds_continuation_sentences(self) -> None:
        example = next(
            tinystories_sentence_pairs(
                [(0, STORY)],
                context_tokens=64,
                response_tokens=128,
            )
        )

        self.assertIn("They kicked the ball together.", example.response)
        self.assertIn("Then they went home.", example.response)


if __name__ == "__main__":
    unittest.main()
