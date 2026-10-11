# EcoRoute benchmark results

66 labelled prompts, simulate mode, baseline = always `openai-gpt-5`.
All impact figures are *estimates* from the methodology in docs/METHODOLOGY.md.

## Complexity classifier

| split | prompts | exact tier | within ±1 tier | under-routed |
|---|---|---|---|---|
| train | 48 | 83% | 100% | 10% |
| holdout | 18 | 56% | 100% | 22% |

Thresholds were calibrated on the *train* split; *holdout* prompts were written afterwards and never used for tuning.

## Savings versus always using the baseline model

| policy | energy (Wh) | baseline (Wh) | energy saved | carbon saved | water saved | cost saved |
|---|---|---|---|---|---|---|
| eco | 2.935 | 21.258 | 86.2% | 95.6% | 89.1% | 90.3% |
| balanced | 2.942 | 21.276 | 86.2% | 95.5% | 89.1% | 90.2% |
| quality | 5.484 | 20.562 | 73.3% | 73.3% | 69.3% | 83.7% |
| cost | 7.291 | 20.730 | 64.8% | 66.8% | 68.8% | 92.2% |

## Models chosen per policy

- **eco**: `groq-gpt-oss-120b` × 18, `groq-llama-3.1-8b` × 16, `groq-gpt-oss-20b` × 16, `mistral-ministral-3b` × 9, `mistral-large-4` × 7
- **balanced**: `groq-gpt-oss-20b` × 25, `groq-gpt-oss-120b` × 18, `groq-llama-3.1-8b` × 16, `mistral-large-4` × 7
- **quality**: `groq-gpt-oss-20b` × 25, `anthropic-claude-haiku-5.5` × 18, `groq-gpt-oss-120b` × 16, `gemini-3.8-flash` × 7
- **cost**: `deepseek-flash` × 18, `groq-llama-3.1-8b` × 16, `local-gpt-oss-20b` × 16, `local-llama3.2-3b` × 9, `mistral-large-4` × 7

## What this means at scale (balanced policy)

Extrapolated to **1 million** prompts with this mix: ~278 kWh, ~114 kg CO2e, ~962 L of water and ~$5,821 saved.
(Per 66 prompts: 1.22 phone charges' worth of energy.)
