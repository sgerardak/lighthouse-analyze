"""The policy's numeric limits as data, and the arithmetic checks over them.

The model reads the policy as prose and is unreliable at comparing a computed
amount against a threshold: in testing it would divide correctly and then state
the opposite conclusion. The limits it is allowed to cite therefore live here as
numbers, and every comparison it claims is recomputed in Python.
"""

from __future__ import annotations

from dataclasses import dataclass

# Amounts are compared in EUR; a cent of slack absorbs float division.
TOLERANCE = 0.01


@dataclass(frozen=True)
class Limit:
    """One numeric threshold, and the policy section that states it."""

    name: str
    value: float
    section: str
    per_person: bool


POLICY_LIMITS: tuple[Limit, ...] = (
    Limit("missing_receipt_manager_approval", 250.0, "3.2", False),
    Limit("no_pre_approval_ceiling", 500.0, "4.1", False),
    Limit("finance_lead_approval_floor", 2500.0, "4.1", False),
    Limit("client_meal_per_person", 80.0, "4.2", True),
    Limit("team_meal_per_person", 40.0, "4.2", True),
    Limit("team_event_total_approval", 500.0, "4.2", False),
    Limit("client_gift_per_person_year", 50.0, "4.4", True),
    Limit("hotel_per_night", 180.0, "5.2", False),
    Limit("hotel_per_night_capital", 250.0, "5.2", False),
    Limit("accrual_reporting_floor", 1000.0, "7.3", False),
)

KNOWN_LIMIT_VALUES = frozenset(limit.value for limit in POLICY_LIMITS)


def _money(value: float) -> str:
    """Format an amount without a pointless .00."""
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def find_limit(value: float, sources: list[str]) -> Limit | None:
    """Find the limit with this value, preferring one the answer cited.

    Several limits share a value (500 EUR appears in both 4.1 and 4.2), so the
    cited sections are used to disambiguate.
    """
    matches = [limit for limit in POLICY_LIMITS if limit.value == value]
    if not matches:
        return None
    for limit in matches:
        if limit.section in sources:
            return limit
    return matches[0]


def verdict_sentence(answer) -> str | None:
    """State the comparison in Python, or None when there is nothing to compare.

    This sentence is prepended to the model's explanation so that the decisive
    numeric claim is computed rather than written: in testing the model would
    report the correct verdict in its fields and the opposite one in its prose.
    It deliberately states only the comparison, never whether the expense is
    allowed overall, because a rule can forbid something that is under its limit
    (gift cards under 50 EUR, for example).
    """
    if answer.verdict == "not_applicable" or answer.limit_applied is None:
        return None

    compared = answer.per_person if answer.per_person is not None else answer.amount_eur
    if compared is None:
        return None

    limit = find_limit(answer.limit_applied, answer.sources)
    unit = " per person" if answer.per_person is not None else ""
    relation = "above" if answer.verdict == "over_limit" else "within"
    section = f" (section {limit.section})" if limit else ""
    return (
        f"{_money(compared)} EUR{unit} is {relation} the "
        f"{_money(answer.limit_applied)} EUR{unit} limit{section}."
    )


def verify_arithmetic(answer) -> list[str]:
    """Recheck the model's own numbers; return a problem per inconsistency.

    An empty list means every claim the model made about amounts agrees with
    what Python computes from the same inputs.
    """
    problems: list[str] = []

    # 1. If the answer divides a total between people, redo the division.
    if answer.amount_eur is not None and answer.headcount:
        expected = answer.amount_eur / answer.headcount
        if answer.per_person is None:
            problems.append(
                f"per_person is missing although amount_eur={answer.amount_eur} "
                f"and headcount={answer.headcount} were given"
            )
        elif abs(expected - answer.per_person) > TOLERANCE:
            problems.append(
                f"per_person={answer.per_person} but "
                f"{answer.amount_eur}/{answer.headcount}={expected:.2f}"
            )

    # 2. The limit must be one the policy actually states.
    if answer.limit_applied is not None and answer.limit_applied not in KNOWN_LIMIT_VALUES:
        problems.append(
            f"limit_applied={answer.limit_applied} is not a limit in the policy"
        )

    # 3. The verdict must follow from the comparison, not from the prose.
    compared = answer.per_person if answer.per_person is not None else answer.amount_eur
    if answer.limit_applied is not None and compared is not None:
        if compared > answer.limit_applied + TOLERANCE:
            expected_verdict = "over_limit"
        else:
            expected_verdict = "within_limit"
        if answer.verdict != expected_verdict:
            problems.append(
                f"verdict={answer.verdict!r} but {compared} vs limit "
                f"{answer.limit_applied} is {expected_verdict!r}"
            )

    return problems
