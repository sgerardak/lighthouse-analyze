"""Measure whether a model gets this policy's numeric thresholds right.

This is the eval behind the model choice argued in docs/memo.md. It grades the
answer a user would actually receive: the tool input is validated and its
arithmetic rechecked exactly as app/service.py does, and the opening sentence is
composed the same way before anything is judged.

Structured fields are graded deterministically. The prose is graded by a Claude
Opus 5 judge against a required conclusion written by hand from the policy.

    python evals/model_eval.py --reps 5 --yes

Run it from the repository root. It calls the API and costs real money; the
estimate is printed and --yes is required before anything is sent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anthropic  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.limits import verdict_sentence, verify_arithmetic  # noqa: E402
from app.policy import get_policy, validate_sources  # noqa: E402
from app.prompts import build_system_prompt, build_user_message  # noqa: E402
from app.schemas import PolicyAnswer  # noqa: E402
from app.service import TOOL_DESCRIPTION, TOOL_NAME  # noqa: E402

SETTINGS = get_settings()
POLICY = get_policy()
BASE_SYSTEM = build_system_prompt(POLICY.text)
JUDGE_MODEL = "claude-opus-5"

# Rough per-million token prices, only for the cost estimate printed up front.
PRICES = {"claude-haiku-4-5-20251001": (1.0, 5.0), "claude-sonnet-5": (2.0, 10.0),
          "claude-opus-5": (5.0, 25.0)}

# A variant that tells the model to check its comparison. It is kept because the
# result is informative: it did not help, and made the hardest case worse.
THRESHOLD_RULE = """
12. When the question involves an amount, find the relevant threshold first,
state whether the amount is above or below it, and only then conclude."""

# (id, question, acceptable section ids, the conclusion the answer must convey)
QUESTIONS = [
    ("q1", "I lost the receipt for a 300 EUR client dinner. What do I do?", ["3.2"],
     "A missing-receipt declaration must be submitted in Brex within 5 business days, "
     "AND because 300 EUR is ABOVE the 250 EUR threshold it also needs the manager's "
     "approval."),
    ("q2", "I lost the receipt for a 40 EUR taxi. Does my manager need to approve the "
           "declaration?", ["3.2"],
     "No manager approval is needed, because 40 EUR is BELOW the 250 EUR threshold. "
     "A declaration is still required."),
    ("q3", "Can I buy a 2,000 EUR standing desk without asking anyone first?", ["4.1"],
     "No. 2,000 EUR falls in the 500-2,500 EUR band, so the manager must approve "
     "before the purchase. The Finance lead's approval is NOT required."),
    ("q4", "I need a 3,000 EUR camera for the marketing team. Who has to approve it?",
     ["4.1"],
     "Because 3,000 EUR is ABOVE 2,500 EUR, BOTH the manager AND the Finance lead "
     "must approve before the purchase."),
    ("q5", "Team dinner for 6 people costing 300 EUR in total. Is that within policy?",
     ["4.2"],
     "No. That is 50 EUR per person, ABOVE the 40 EUR per person limit for team meals."),
    ("q6", "Client lunch for 4 people, 280 EUR in total. Is that within policy?", ["4.2"],
     "Yes. That is 70 EUR per person, BELOW the 80 EUR per person limit for client meals."),
    ("q7", "Can I book a hotel at 220 EUR per night in a city that is not a capital "
           "city?", ["5.2"],
     "No. For non-capital cities the limit is 180 EUR per night, and 220 EUR is ABOVE it."),
    ("q8", "Can I book a hotel at 220 EUR per night in a capital city?", ["5.2"],
     "Yes. In capital cities the limit is 250 EUR per night, and 220 EUR is BELOW it."),
    ("q9", "I want to give a client a gift worth 60 EUR. Is that allowed?", ["4.4"],
     "No. Client gifts are limited to 50 EUR per person per year, and 60 EUR is ABOVE that."),
    ("q10", "Can I give a client a 30 EUR gift card?", ["4.4"],
     "No. Gift cards are NEVER allowed as gifts, regardless of the amount, even though "
     "30 EUR is below the 50 EUR limit."),
    ("q11", "We received 800 EUR of consulting services this month but no invoice yet. "
            "Do I need to tell Finance for an accrual?", ["7.3"],
     "No. Accruals are reported when the expected amount is ABOVE 1,000 EUR, and 800 EUR "
     "is BELOW that threshold."),
    ("q12", "How many holiday days do I get per year?", [], "OUT_OF_SCOPE"),
]

CONFIGS = {
    "shipped": dict(model=SETTINGS.MODEL_NAME, system=BASE_SYSTEM, thinking=None),
    "threshold-rule": dict(model=SETTINGS.MODEL_NAME, system=BASE_SYSTEM + THRESHOLD_RULE,
                           thinking=None),
    "thinking": dict(model=SETTINGS.MODEL_NAME, system=BASE_SYSTEM, thinking=2000),
    "sonnet5": dict(model="claude-sonnet-5", system=BASE_SYSTEM, thinking=None),
}

client = anthropic.AsyncAnthropic(api_key=SETTINGS.ANTHROPIC_API_KEY, timeout=60,
                                  max_retries=2)
limit = asyncio.Semaphore(4)

JUDGE_TOOL = {
    "name": "grade",
    "description": "Record the grade for one candidate answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["pass", "fail"]},
            "reason": {"type": "string", "description": "One short sentence."},
        },
        "required": ["verdict", "reason"],
    },
}


async def ask(config: dict, question: str) -> dict:
    """One call, graded the way the service would grade it."""
    schema = PolicyAnswer.model_json_schema()
    tool = {"name": TOOL_NAME, "description": TOOL_DESCRIPTION,
            "input_schema": {**schema, "additionalProperties": False}, "strict": True}
    request = {
        "model": config["model"],
        "max_tokens": 4096 if config["thinking"] else SETTINGS.MAX_OUTPUT_TOKENS,
        "system": [{"type": "text", "text": config["system"],
                    "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": build_user_message(question)}],
        "tools": [tool],
    }
    if config["thinking"]:
        request["thinking"] = {"type": "enabled", "budget_tokens": config["thinking"]}
        # Forced tool choice and extended thinking cannot be combined.
        request["tool_choice"] = {"type": "auto"}
    else:
        request["tool_choice"] = {"type": "tool", "name": TOOL_NAME}

    row = {"error": None, "latency_ms": 0, "in": 0, "out": 0, "cache_read": 0,
           "sources_ok": False, "answer": None}

    started = time.perf_counter()
    async with limit:
        try:
            response = await client.messages.create(**request)
        except Exception as exc:  # noqa: BLE001 - the eval reports, never raises
            row["error"] = f"{type(exc).__name__}: {str(exc)[:80]}"
            return row
    row["latency_ms"] = int((time.perf_counter() - started) * 1000)

    usage = response.usage
    row["in"] = usage.input_tokens
    row["out"] = usage.output_tokens
    row["cache_read"] = getattr(usage, "cache_read_input_tokens", 0) or 0

    block = next((b for b in response.content if b.type == "tool_use"), None)
    if block is None:
        row["error"] = f"no_tool_use (stop={response.stop_reason})"
        return row

    try:
        answer = PolicyAnswer.model_validate(block.input)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"schema: {str(exc).splitlines()[1].strip()[:70]}"
        return row

    unknown = validate_sources(answer.sources, POLICY)
    if unknown:
        row["error"] = f"invented citation: {', '.join(unknown)}"
        return row

    problems = verify_arithmetic(answer)
    if problems:
        row["error"] = f"arithmetic: {problems[0][:70]}"
        return row

    # Grade what a user would read, sentence included.
    sentence = verdict_sentence(answer)
    row["answer"] = f"{sentence} {answer.answer}" if sentence else answer.answer
    row["parsed"] = answer
    return row


async def judge(question: str, required: str, candidate: str) -> tuple[str, str]:
    """Grade one answer against the conclusion the policy requires."""
    async with limit:
        response = await client.messages.create(
            model=JUDGE_MODEL, max_tokens=1000,
            system="You are a strict grader for a policy assistant. Use the grade tool.",
            messages=[{"role": "user", "content":
                       f"Employee question:\n{question}\n\n"
                       f"REQUIRED CONCLUSION (ground truth from the policy):\n{required}\n\n"
                       f"CANDIDATE ANSWER:\n{candidate}\n\n"
                       "Does the candidate convey the required conclusion without "
                       "contradicting it? Ignore wording, tone and extra correct detail. "
                       "Fail it if the numeric comparison or the conclusion is wrong, "
                       "missing or reversed."}],
            tools=[JUDGE_TOOL], tool_choice={"type": "tool", "name": "grade"})
    graded = next(b for b in response.content if b.type == "tool_use")
    return graded.input["verdict"], graded.input["reason"]


async def run_config(name: str, config: dict, reps: int) -> list[dict]:
    """Run every question `reps` times and grade the results."""
    plan = [(q, rep) for q in QUESTIONS for rep in range(reps)]
    rows = await asyncio.gather(*[ask(config, q[1]) for q, _ in plan])

    pending = []
    for ((qid, question, sources, truth), rep), row in zip(plan, rows):
        row.update(config=name, qid=qid, rep=rep, verdict=None, reason=None)
        answer = row.pop("parsed", None)
        if answer is None:
            continue
        if truth == "OUT_OF_SCOPE":
            ok = not answer.in_scope and answer.escalate_to_finance and not answer.sources
            row["sources_ok"] = ok
            row["verdict"] = "pass" if ok else "fail"
            row["reason"] = "" if ok else f"in_scope={answer.in_scope}"
        else:
            row["sources_ok"] = any(s in answer.sources for s in sources)
            pending.append((row, question, truth))

    graded = await asyncio.gather(*[judge(q, t, r["answer"]) for r, q, t in pending])
    for (row, _, _), (verdict, reason) in zip(pending, graded):
        row["verdict"], row["reason"] = verdict, reason
    return rows


def report(rows: list[dict], configs: list[str], reps: int) -> None:
    print("\n" + "=" * 96)
    print(f"{'config':<16}{'answers correct':>17}{'citations ok':>15}"
          f"{'usable output':>16}{'p50 ms':>9}{'tok in/out':>13}")
    print("-" * 96)
    for name in configs:
        subset = [r for r in rows if r["config"] == name]
        total = len(subset)
        usable = [r for r in subset if r["error"] is None]
        correct = [r for r in subset if r["verdict"] == "pass"]
        cited = [r for r in subset if r["sources_ok"]]
        latencies = [r["latency_ms"] for r in usable] or [0]
        print(f"{name:<16}{len(correct):>8}/{total:<3}({100 * len(correct) // total:>3}%)"
              f"{len(cited):>7}/{total:<3}({100 * len(cited) // total:>3}%)"
              f"{len(usable):>8}/{total:<3}({100 * len(usable) // total:>3}%)"
              f"{int(statistics.median(latencies)):>9}"
              f"{int(statistics.mean([r['in'] for r in usable] or [0])):>7}/"
              f"{int(statistics.mean([r['out'] for r in usable] or [0])):<5}")

    print("\n" + "=" * 96)
    print(f"PER QUESTION (correct out of {reps})")
    print("-" * 96)
    print(f"{'':<6}" + "".join(f"{c:>16}" for c in configs))
    for qid, question, _, _ in QUESTIONS:
        cells = []
        for name in configs:
            subset = [r for r in rows if r["config"] == name and r["qid"] == qid]
            cells.append(f"{sum(1 for r in subset if r['verdict'] == 'pass')}/{len(subset)}")
        print(f"{qid:<6}" + "".join(f"{c:>16}" for c in cells) + f"  {question[:34]}")

    failures = [r for r in rows if r["verdict"] != "pass"]
    if failures:
        print("\n" + "=" * 96)
        print("FAILURES (first per config and question)")
        print("-" * 96)
        seen = set()
        for row in failures:
            key = (row["config"], row["qid"])
            if key in seen:
                continue
            seen.add(key)
            print(f"  [{row['config']:<15}] {row['qid']:<4} "
                  f"{str(row['error'] or row['reason'])[:95]}")


def estimate(configs: list[str], reps: int) -> float:
    """Rough dollar estimate, so nobody starts a run blind."""
    calls = len(QUESTIONS) * reps
    total = 0.0
    for name in configs:
        model = CONFIGS[name]["model"]
        price_in, price_out = PRICES.get(model, (2.0, 10.0))
        total += calls * (3400 * price_in + 400 * price_out) / 1e6
    judge_in, judge_out = PRICES[JUDGE_MODEL]
    total += calls * len(configs) * (700 * judge_in + 120 * judge_out) / 1e6
    return total


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--configs", default="shipped",
                        help="comma-separated: " + ", ".join(CONFIGS) + ", or 'all'")
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    parser.add_argument("--out", default="evals/results.json")
    args = parser.parse_args()

    configs = list(CONFIGS) if args.configs == "all" else args.configs.split(",")
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        parser.error(f"unknown config(s): {', '.join(unknown)}")

    calls = len(QUESTIONS) * args.reps * len(configs)
    print(f"{calls} answer calls plus {calls} judge calls "
          f"across {len(configs)} config(s), {args.reps} reps.")
    print(f"Rough cost: ${estimate(configs, args.reps):.2f}")
    if not args.yes:
        print("Re-run with --yes to spend it.")
        return

    rows: list[dict] = []
    for name in configs:
        print(f"running {name} ...", flush=True)
        rows += await run_config(name, CONFIGS[name], args.reps)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False, default=str)
    report(rows, configs, args.reps)
    print(f"\nrows written to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
