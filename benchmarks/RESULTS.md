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
| eco | 7.128 | 21.200 | 66.4% | 66.4% | 66.2% | 64.9% |
| balanced | 7.221 | 21.292 | 66.1% | 66.1% | 63.6% | 64.6% |
| quality | 9.587 | 21.484 | 55.4% | 55.4% | 54.0% | 46.2% |
| cost | 10.325 | 21.622 | 52.2% | 48.6% | 46.7% | 64.7% |

## Models chosen per policy

- **eco**: `groq-gpt-oss-20b` × 25, `groq-gpt-oss-120b` × 18, `groq-llama-3.1-8b` × 16, `openai-gpt-5` × 7
- **balanced**: `groq-gpt-oss-20b` × 25, `groq-gpt-oss-120b` × 18, `groq-llama-3.1-8b` × 16, `openai-gpt-5` × 4, `gemini-2.5-pro` × 3
- **quality**: `groq-gpt-oss-120b` × 34, `groq-gpt-oss-20b` × 25, `anthropic-claude-sonnet-4.5` × 7
- **cost**: `groq-gpt-oss-120b` × 18, `groq-llama-3.1-8b` × 16, `local-qwen2.5-7b` × 16, `local-llama3.2-3b` × 9, `gemini-2.5-pro` × 7

## What this means at scale (balanced policy)

Extrapolated to **1 million** prompts with this mix: ~213 kWh, ~79 kg CO2e, ~687 L of water and ~$4,173 saved.
(Per 66 prompts: 0.94 phone charges' worth of energy.)
