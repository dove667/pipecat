#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the instruction that teaches an LLM to write expression tags."""

from pipecat.expressive.expression import ExpressionCapabilities, ExpressionTerm
from pipecat.expressive.instructions import expression_instructions, unperformable_tags
from pipecat.expressive.perception import PerceptionCapabilities, PerceptionTerm
from pipecat.expressive.policy import ExpressionGuidance, ExpressionPolicy

FIXED = ExpressionCapabilities(
    events=[ExpressionTerm("laughter")],
    modes=[ExpressionTerm("calm"), ExpressionTerm("happy"), ExpressionTerm("sad")],
    pause_range_ms=(100, 5000),
)

FREE_TEXT = ExpressionCapabilities(
    events=[ExpressionTerm("laughs"), ExpressionTerm("woo", "an excited whoop")],
    modes=[ExpressionTerm("excited"), ExpressionTerm("whispers", "speaking in a whisper")],
    free_text_events=True,
    free_text_modes=True,
    emphasis=True,
)


def test_nothing_is_taught_for_a_service_that_performs_nothing():
    assert expression_instructions(None) is None
    assert expression_instructions(ExpressionCapabilities()) is None


def test_a_fixed_list_is_offered_as_the_only_choices():
    instructions = expression_instructions(FIXED)

    assert "Use only these tones, written exactly as listed: calm, happy, sad." in instructions
    assert "Use only these sounds, written exactly as listed: [[laughter]]." in instructions


def test_a_free_text_list_is_offered_as_examples():
    instructions = expression_instructions(FREE_TEXT)

    assert "For example: excited, whispers (speaking in a whisper). Others" in instructions
    assert "For example: [[laughs]], [[woo]] (an excited whoop). Others" in instructions
    assert "exactly as listed" not in instructions


def test_only_the_controls_a_service_performs_are_taught():
    fixed = expression_instructions(FIXED)
    free_text = expression_instructions(FREE_TEXT)

    assert "^^word^^" not in fixed
    assert "[[pause 500]]" in fixed
    assert "^^word^^ stresses a word." in free_text
    assert "[[pause" not in free_text


def test_a_policy_narrows_a_fixed_list_in_its_own_order():
    policy = ExpressionPolicy(modes=["sad", "relieved", "calm"], events=[])

    instructions = expression_instructions(FIXED, policy)

    assert "exactly as listed: sad, calm." in instructions
    assert "[[sound]]" not in instructions


def test_a_policy_names_any_mode_for_a_free_text_service():
    policy = ExpressionPolicy(modes=["whispers", "relieved"])

    instructions = expression_instructions(FREE_TEXT, policy)

    assert "For example: whispers (speaking in a whisper), relieved." in instructions


def test_nothing_is_taught_when_a_policy_offers_nothing_performable():
    capabilities = ExpressionCapabilities(modes=[ExpressionTerm("calm")])
    policy = ExpressionPolicy(guidance=[ExpressionGuidance.MATCH_STYLE], modes=["relieved"])

    assert expression_instructions(capabilities, policy) is None


def test_guidance_follows_the_tags_in_order():
    policy = ExpressionPolicy(
        guidance=[ExpressionGuidance.MATCH_STYLE, "Never laugh while discussing billing."]
    )

    instructions = expression_instructions(FIXED, policy)

    assert instructions.endswith(
        f"\n\n{ExpressionGuidance.MATCH_STYLE}\nNever laugh while discussing billing."
    )


def test_unperformable_tags_are_those_a_fixed_list_lacks():
    policy = ExpressionPolicy(modes=["calm", "relieved"], events=["laughter", "sighs"])

    assert unperformable_tags(FIXED, policy) == ["<<relieved>>", "[[sighs]]"]
    assert unperformable_tags(FREE_TEXT, policy) == []


PERCEPTION = PerceptionCapabilities(
    signals=[
        PerceptionTerm("tone", labels=["positive", "neutral", "sad", "frustrated"]),
        PerceptionTerm("speech_rate", "Words per second."),
    ]
)


def test_perception_is_taught_with_an_example_and_its_signals():
    instructions = expression_instructions(None, perception=PERCEPTION)

    assert "such as [heard: tone positive 0.86, neutral 0.10]," in instructions
    assert "A reading below 0.5 is a guess: don't act on it." in instructions
    assert instructions.endswith(
        "- tone: positive, neutral, sad, frustrated.\n- speech_rate: Words per second."
    )


def test_perception_comes_before_the_tags_and_the_guidance_last():
    policy = ExpressionPolicy(guidance=[ExpressionGuidance.MATCH_STYLE])

    instructions = expression_instructions(FIXED, policy, perception=PERCEPTION)

    reading = instructions.index("[heard:")
    tags = instructions.index("You can direct how your words are spoken")
    assert reading < tags < instructions.index(ExpressionGuidance.MATCH_STYLE)


def test_guidance_is_kept_for_perception_without_tags():
    policy = ExpressionPolicy(guidance=["If the user sounds frustrated, apologize once."])

    instructions = expression_instructions(None, policy, perception=PERCEPTION)

    assert instructions.endswith("\n\nIf the user sounds frustrated, apologize once.")
