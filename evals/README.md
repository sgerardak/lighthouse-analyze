# Eval

The measurement behind the model choice in [../docs/memo.md](../docs/memo.md).

Twelve policy questions with known-correct answers, nine of them numeric
threshold comparisons, run N times against one or more configurations. It grades
the answer a user would actually receive: the tool input is validated and its
arithmetic rechecked exactly as `app/service.py` does, and the opening sentence
is composed the same way before anything is judged. Grading the model's raw
prose instead would measure something the service never shows anyone.

Three things are scored separately, because they fail for different reasons:

| Column | Meaning |
|---|---|
| answers correct | A Claude Opus 5 judge, against a conclusion written by hand from the policy |
| citations ok | Did it cite a section that actually answers the question |
| usable output | Did it survive the schema, citation and arithmetic checks at all |

## Running it

```bash
python evals/model_eval.py                       # prints the cost, sends nothing
python evals/model_eval.py --yes                 # the shipped config, 5 reps
python evals/model_eval.py --configs all --yes   # every config
python evals/model_eval.py --configs shipped,sonnet5 --reps 3 --yes
```

From the repository root, with `.env` populated. **It calls the API and costs
real money.** Nothing is sent until `--yes` is passed; without it the run prints
the planned call count and a rough dollar estimate and stops. A full
`--configs all --reps 5` run is 240 answer calls plus judging; the script
estimates $3.18 for it, and the equivalent run during development cost about
$2.50. The estimator rounds up on purpose.

Per-row results land in `evals/results.json`, which is gitignored.

## Configurations

| Name | What it tests |
|---|---|
| `shipped` | Exactly what the service runs today |
| `threshold-rule` | Adds a prompt rule telling the model to check its comparison |
| `thinking` | Extended thinking, which forces `tool_choice: auto` |
| `sonnet5` | A larger model, same prompt |

`threshold-rule` is kept because it is the most useful negative result in the
exercise: it did not help overall and made the hardest question worse.

## Reading the numbers honestly

At 5 reps each cell is n=5 and each config total is n=60. That is enough to see
a question fail four times out of five, and **not** enough to call 88% different
from 92%. The per-question column carries the signal; the headline percentage
mostly does not.

The judge is also unvalidated — it has never been checked against human labels.
Spot-reading its stated reasons is worth doing before trusting a close result.
