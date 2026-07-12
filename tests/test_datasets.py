"""Tests for evaluation dataset row conversion."""

from corag.evaluation.datasets import (
    example_from_2wiki,
    example_from_hotpotqa,
    extract_supporting_titles,
)


def test_extract_supporting_titles_hotpotqa_columnar_format():
    # HuggingFace hotpot_qa distractor encodes supporting_facts columnar
    supporting_facts = {
        "title": ["Scott Derrickson", "Ed Wood", "Scott Derrickson"],
        "sent_id": [0, 0, 1],
    }

    titles = extract_supporting_titles(supporting_facts)

    # Full titles, deduplicated, order preserved — not first characters
    assert titles == ["Scott Derrickson", "Ed Wood"]


def test_extract_supporting_titles_2wiki_pair_format():
    supporting_facts = [["Albert Einstein", 0], ["Ulm", 2], ["Albert Einstein", 1]]

    titles = extract_supporting_titles(supporting_facts)

    assert titles == ["Albert Einstein", "Ulm"]


def test_extract_supporting_titles_handles_missing_or_empty():
    assert extract_supporting_titles(None) == []
    assert extract_supporting_titles({}) == []
    assert extract_supporting_titles([]) == []


def test_example_from_hotpotqa():
    item = {
        "id": "5a8b57f25542995d1e6f1371",
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "yes",
        "supporting_facts": {
            "title": ["Scott Derrickson", "Ed Wood"],
            "sent_id": [0, 0],
        },
        "type": "comparison",
        "level": "hard",
    }

    example = example_from_hotpotqa(item)

    assert example.id == "5a8b57f25542995d1e6f1371"
    assert example.answer == "yes"
    assert example.supporting_facts == ["Scott Derrickson", "Ed Wood"]
    assert example.type == "comparison"
    assert example.level == "hard"


def test_example_from_2wiki():
    item = {
        "_id": "8813f87c0bdd11eba7f7acde48001122",
        "question": "Where was the father of Albert Einstein born?",
        "answer": "Buchau",
        "supporting_facts": [["Albert Einstein", 0], ["Hermann Einstein", 1]],
        "type": "bridge",
    }

    example = example_from_2wiki(item)

    assert example.id == "8813f87c0bdd11eba7f7acde48001122"
    assert example.supporting_facts == ["Albert Einstein", "Hermann Einstein"]
    assert example.type == "bridge"
