"""Tests for the deterministic arithmetic checks."""

import pytest

from app.limits import (
    KNOWN_LIMIT_VALUES,
    POLICY_LIMITS,
    verdict_sentence,
    verify_arithmetic,
)
from app.policy import get_policy
from app.schemas import PolicyAnswer


def answer(**overrides) -> PolicyAnswer:
    """Build a valid PolicyAnswer, overriding the fields under test."""
    fields = dict(
        answer="Team meals are allowed up to 40 EUR per person.",
        sources=["4.2"],
        in_scope=True,
        escalate_to_finance=False,
        confidence="high",
        amount_eur=300.0,
        headcount=6,
        per_person=50.0,
        limit_applied=40.0,
        verdict="above_threshold",
    )
    fields.update(overrides)
    return PolicyAnswer(**fields)


def test_every_limit_cites_a_real_policy_section():
    """The limits table must stay in step with the policy document."""
    sections = get_policy().sections
    for limit in POLICY_LIMITS:
        assert limit.section in sections, f"{limit.name} cites missing {limit.section}"


def test_consistent_answer_passes():
    assert verify_arithmetic(answer()) == []


def test_reversed_verdict_is_caught():
    """The observed failure: right division, opposite conclusion."""
    problems = verify_arithmetic(answer(verdict="at_or_below_threshold"))

    assert len(problems) == 1
    assert "at_or_below_threshold" in problems[0] and "above_threshold" in problems[0]


def test_wrong_division_is_caught():
    problems = verify_arithmetic(answer(per_person=40.0, verdict="at_or_below_threshold"))

    assert any("300.0/6=50.00" in p for p in problems)


def test_missing_per_person_is_caught():
    problems = verify_arithmetic(answer(per_person=None, verdict="not_applicable"))

    assert any("per_person is missing" in p for p in problems)


def test_invented_limit_is_caught():
    """A limit that appears nowhere in the policy is a fabrication."""
    problems = verify_arithmetic(answer(limit_applied=45.0))

    assert any("not a limit in the policy" in p for p in problems)
    assert 45.0 not in KNOWN_LIMIT_VALUES


def test_amount_without_headcount_is_compared_directly():
    """Non-per-person limits compare the total, not a per-person figure."""
    over = answer(amount_eur=300.0, headcount=None, per_person=None,
                  limit_applied=250.0, verdict="above_threshold")
    assert verify_arithmetic(over) == []

    under = answer(amount_eur=200.0, headcount=None, per_person=None,
                   limit_applied=250.0, verdict="at_or_below_threshold")
    assert verify_arithmetic(under) == []


def test_exactly_at_the_limit_is_within():
    """The policy says 'up to 40 EUR', so 40 is allowed."""
    at_limit = answer(amount_eur=240.0, headcount=6, per_person=40.0,
                      limit_applied=40.0, verdict="at_or_below_threshold")

    assert verify_arithmetic(at_limit) == []


def test_questions_without_amounts_are_left_alone():
    """A non-numeric answer has nothing to verify."""
    plain = answer(amount_eur=None, headcount=None, per_person=None,
                   limit_applied=None, verdict="not_applicable")

    assert verify_arithmetic(plain) == []


def test_headcount_must_be_positive():
    """A zero headcount would divide by zero, so the schema rejects it."""
    with pytest.raises(ValueError):
        answer(headcount=0)


def test_verdict_sentence_states_a_per_person_overage():
    assert verdict_sentence(answer()) == (
        "50 EUR per person is above the 40 EUR per person limit (section 4.2)."
    )


def test_verdict_sentence_states_a_total_at_or_below_threshold():
    within = answer(amount_eur=200.0, headcount=None, per_person=None,
                    limit_applied=250.0, verdict="at_or_below_threshold", sources=["3.2"])

    assert verdict_sentence(within) == (
        "200 EUR is below the 250 EUR threshold (section 3.2)."
    )


def test_verdict_sentence_disambiguates_shared_values_by_citation():
    """500 EUR appears in both 4.1 and 4.2; the cited section decides."""
    event = answer(amount_eur=600.0, headcount=None, per_person=None,
                   limit_applied=500.0, verdict="above_threshold", sources=["4.2"])
    purchase = event.model_copy(update={"sources": ["4.1"]})

    assert "section 4.2" in verdict_sentence(event)
    assert "section 4.1" in verdict_sentence(purchase)


def test_no_sentence_when_no_limit_is_involved():
    plain = answer(amount_eur=None, headcount=None, per_person=None,
                   limit_applied=None, verdict="not_applicable")

    assert verdict_sentence(plain) is None


def test_sentence_keeps_cents_when_they_matter():
    odd = answer(amount_eur=250.0, headcount=3, per_person=250.0 / 3,
                 limit_applied=80.0, verdict="at_or_below_threshold", sources=["4.2"])

    assert verdict_sentence(odd).startswith("83.33 EUR per person is within")


def test_trigger_thresholds_are_not_described_as_limits():
    """Crossing 250 EUR in 3.2 means needing approval, not breaking a rule."""
    receipt = answer(amount_eur=300.0, headcount=None, per_person=None,
                     limit_applied=250.0, verdict="above_threshold", sources=["3.2"])

    sentence = verdict_sentence(receipt)

    assert sentence == "300 EUR is above the 250 EUR threshold (section 3.2)."
    assert "limit" not in sentence
    assert verify_arithmetic(receipt) == []


def test_caps_are_still_described_as_limits():
    """5.2's 250 EUR shares a value with 3.2's trigger but is a real cap."""
    hotel = answer(amount_eur=300.0, headcount=None, per_person=None,
                   limit_applied=250.0, verdict="above_threshold", sources=["5.2"])

    assert verdict_sentence(hotel) == (
        "300 EUR is above the 250 EUR limit (section 5.2)."
    )


def test_below_a_trigger_reads_as_below_not_within():
    accrual = answer(amount_eur=800.0, headcount=None, per_person=None,
                     limit_applied=1000.0, verdict="at_or_below_threshold",
                     sources=["7.3"])

    assert verdict_sentence(accrual) == (
        "800 EUR is below the 1,000 EUR threshold (section 7.3)."
    )
