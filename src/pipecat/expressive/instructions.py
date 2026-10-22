#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The instruction that teaches an LLM to take part in expressive conversation.

Part of expressive conversation (alpha). The instruction has three parts: how
to read what was perceived of how the user spoke, built from a perception
producer's :class:`~pipecat.expressive.perception.PerceptionCapabilities`; the
tags the active TTS service can perform, built from its
:class:`~pipecat.expressive.expression.ExpressionCapabilities`; and the
developer's guidance on how to answer, from an
:class:`~pipecat.expressive.policy.ExpressionPolicy`.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

from pipecat.expressive.expression import ExpressionCapabilities, ExpressionTerm
from pipecat.expressive.perception import (
    PerceptionCapabilities,
    PerceptionSignal,
    PerceptionTerm,
    render_perception,
)
from pipecat.expressive.policy import ExpressionPolicy


def expression_instructions(
    capabilities: ExpressionCapabilities | None,
    policy: ExpressionPolicy | None = None,
    *,
    perception: PerceptionCapabilities | None = None,
) -> str | None:
    """Build the instruction for an LLM in an expressive conversation.

    Args:
        capabilities: What the TTS service that speaks the LLM's replies can
            perform, or None for nothing.
        policy: Which tags to offer, and guidance on how to answer.
        perception: The signals perceived of how the user spoke, or None when
            nothing is perceived.

    Returns:
        The instruction, or None when nothing is perceived and the TTS service
        can perform none of the tags offered.
    """
    policy = policy or ExpressionPolicy()
    parts = [
        part
        for part in (_perception_section(perception), _tags_section(capabilities, policy))
        if part
    ]
    if not parts:
        return None
    if policy.guidance:
        parts.append("\n".join(policy.guidance))
    return "\n\n".join(parts)


def _perception_section(perception: PerceptionCapabilities | None) -> str | None:
    if not perception or not perception.signals:
        return None
    such_as = ""
    example = next((term for term in perception.signals if len(term.labels) >= 2), None)
    if example:
        labels = [(example.labels[0], 0.86), (example.labels[1], 0.10)]
        such_as = f" such as {render_perception([PerceptionSignal(example.name, labels)])},"
    signals = "\n".join(_signal_line(term) for term in perception.signals)
    return (
        f"A user message may start with how the user sounded,{such_as} each signal with its "
        "likeliest labels and how sure the reading is, from 0 to 1, or with a measure. The "
        "reading can be wrong, so weigh it against their words. A reading below 0.5 is a "
        f"guess: don't act on it. Never write it yourself. The signals are:\n{signals}"
    )


def _signal_line(term: PerceptionTerm) -> str:
    if not term.labels:
        return f"- {term.name}: {(term.description or 'a measure').rstrip('.')}."
    line = f"- {term.name}: {', '.join(term.labels)}."
    return f"{line} {term.description}" if term.description else line


def _tags_section(
    capabilities: ExpressionCapabilities | None, policy: ExpressionPolicy
) -> str | None:
    if capabilities is None:
        return None
    tags = []
    modes = _offered(capabilities.modes, policy.modes, capabilities.free_text_modes)
    if modes:
        tags.append(
            "- <<tone>> at the start of a sentence speaks it, and the sentences after it, in "
            "that tone, until the next tone tag. <<neutral>> returns to your normal voice. "
            "A tone is one word. " + _choices("tones", modes, capabilities.free_text_modes)
        )
    events = _offered(capabilities.events, policy.events, capabilities.free_text_events)
    if events:
        tags.append(
            "- [[sound]] makes a sound at that point. "
            + _choices("sounds", events, capabilities.free_text_events, brackets=True)
        )
    if capabilities.emphasis:
        tags.append("- ^^word^^ stresses a word.")
    if capabilities.pause_range_ms:
        tags.append("- [[pause 500]] pauses for about that many milliseconds.")
    if not tags:
        return None
    return "You can direct how your words are spoken with tags:\n" + "\n".join(tags)


def unperformable_tags(capabilities: ExpressionCapabilities, policy: ExpressionPolicy) -> list[str]:
    """The tags a policy offers that a TTS service can't perform.

    Args:
        capabilities: What the TTS service can perform.
        policy: The policy to check.

    Returns:
        Each such tag as it would be written, such as ``"<<relieved>>"``.
    """
    tags = [
        f"<<{name}>>"
        for name in policy.modes or []
        if not _performs(name, capabilities.modes, capabilities.free_text_modes)
    ]
    tags += [
        f"[[{name}]]"
        for name in policy.events or []
        if not _performs(name, capabilities.events, capabilities.free_text_events)
    ]
    return tags


def _performs(name: str, declared: list[ExpressionTerm], free_text: bool) -> bool:
    return free_text or any(term.name == name for term in declared)


def _offered(
    declared: list[ExpressionTerm], names: list[str] | None, free_text: bool
) -> list[ExpressionTerm]:
    """The terms to offer: those declared, narrowed to the policy's names."""
    if names is None:
        return declared
    by_name = {term.name: term for term in declared}
    if free_text:
        return [by_name.get(name, ExpressionTerm(name)) for name in names]
    return [by_name[name] for name in names if name in by_name]


def _choices(
    kind: str, terms: list[ExpressionTerm], free_text: bool, *, brackets: bool = False
) -> str:
    def term_text(term: ExpressionTerm) -> str:
        name = f"[[{term.name}]]" if brackets else term.name
        return f"{name} ({term.description})" if term.description else name

    listed = ", ".join(term_text(term) for term in terms)
    if free_text:
        return f"For example: {listed}. Others work too."
    return (
        f"Use only these {kind}, written exactly as listed: {listed}. "
        "Any other word in a tag is ignored, so if none fits, write no tag."
    )
