"""Dataset adapters for Gator training and evaluation."""

from .smoltalk import (
    AssistantExample,
    Conversation,
    DialogueTurn,
    SmolTalkStatistics,
    assistant_examples,
    ascii_letter_ratio,
    conversation_ascii_letter_ratio,
    format_context,
    iter_jsonl_conversations,
    load_smoltalk,
    parse_dialogue,
    split_conversations,
    summarize_smoltalk,
)
from .tinystories import (
    TINYSTORIES_SOURCE,
    iter_tinystories,
    split_story_sentences,
    tinystories_conversations,
    tinystories_sentence_pairs,
)

__all__ = [
    "AssistantExample",
    "Conversation",
    "DialogueTurn",
    "SmolTalkStatistics",
    "assistant_examples",
    "ascii_letter_ratio",
    "conversation_ascii_letter_ratio",
    "format_context",
    "iter_jsonl_conversations",
    "load_smoltalk",
    "parse_dialogue",
    "split_conversations",
    "summarize_smoltalk",
    "TINYSTORIES_SOURCE",
    "iter_tinystories",
    "split_story_sentences",
    "tinystories_conversations",
    "tinystories_sentence_pairs",
]
