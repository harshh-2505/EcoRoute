# Methodology

EcoRoute makes three kinds of estimate: **how hard a prompt is**, **how much energy, carbon, water and
money a model would spend answering it**, and **which model gives the best trade-off**. Every constant
below lives in code or YAML and can be changed; none of it is presented as ground truth.

## 1. Prompt complexity (`ecoroute/complexity.py`)

The analyser extracts interpretable features from the latest user message:

| Feature | Examples of what it detects | Max contribution |
|---|---|---|
| length | token estimate beyond the first 8 tokens | +0.25 |
| reasoning | *why, explain, analyse, compare, trade-offs, step by step* | +0.30 |
| deep reasoning | *prove, derive, rigorous, theorem, edge cases* | +0.35 |
| code | language names, *implement, debug, refactor*, code blocks | +0.35 |
| maths | *integral, probability, solve*, symbols such as ∫ or `x^2` | +0.25 |
| system design | *architecture, distributed, scalable, fault-tolerant* | +0.40 |
| expert domain | legal, clinical, financial modelling, research | +0.25 |
| long-form / requested length | *essay, report*, "1500 words" | +0.25 / +0.35 |
| structure | multiple questions, enumerations, clauses, constraints, depth words | small |
| factoid / brevity | *what is, translate, capital of, briefly* | **negative** |
| greeting | "hi", "thanks" | −0.50 |

Each feature uses a saturating transform `n / (n + k)` so the tenth keyword matters less than the
first. The sum (plus a 0.04 base) is clipped to 0–1 and cut into tiers at **0.12 / 0.20 / 0.48 / 0.74**.
Hard floors stop obviously technical prompts from reaching tiny models (any code → tier ≥ 2; a proof
→ tier ≥ 3; heavy system-design or proof language → tier ≥ 4).

**Calibration and evaluation.** The thresholds were grid-searched on the 48-prompt *train* split of
`benchmarks/prompts.jsonl`. Eighteen *holdout* prompts were written afterwards and never used for
tuning. Current results: train 83% exact / 100% within one tier; holdout 56% exact / 100% within one
tier. Under-routing (predicting a lower tier than the label) happens on 22% of holdout prompts, which
is why the router keeps a fallback chain and why `min_tier` exists as a per-request override.

The expected answer length per tier (60 / 220 / 500 / 900 / 1600 tokens, or 1.35× an explicit word
target) is used to *estimate* impact before a call; the ledger always records the **real** token
counts returned by the provider.

## 2. Energy per token (`ecoroute/impact.py`)

Energy is expressed as **Wh per 1,000 output tokens**, from up to three signals.

### a) Compute model — open-weight models
A transformer forward pass costs about `2 × N_active` FLOPs per generated token. With an *effective*
delivered efficiency of `2×10¹¹ FLOP/J` (batched decoding on H100-class hardware is memory-bound, so
this is far below peak) and the data centre's PUE:

```
Wh/1k = 2 · N_active · 1000 / 2e11 / 3600 × PUE
```

Llama 3.1 8B → ≈0.025 Wh/1k; Llama 3.3 70B → ≈0.22 Wh/1k. Mixture-of-experts models such as
gpt-oss use *active* parameters (3.6 B / 5.1 B), which is why they come out so efficient.

### b) Price proxy — closed models
Providers do not publish parameter counts, but list prices are driven largely by GPU-time, and
GPU-time is energy. EcoRoute maps output price to energy with a sub-linear power law:

```
Wh/1k = 0.5 × (price_out / $10)^0.6
```

The anchor (a $10/M-token frontier model ≈ 0.5 Wh per 1k output tokens) is consistent with public
per-prompt disclosures in the 0.24–0.34 Wh range for typical chatbot prompts. The exponent is below 1
because prices also include margin, research costs and demand effects. As a cross-check, the formula
gives 0.028 Wh/1k at $0.08 (compute model for an 8B model: 0.025) and 0.11 at $0.79 (compute model for
70B: 0.22) — the same order of magnitude from two independent signals.

For open-weight models both signals exist; EcoRoute takes a weighted geometric mean (80% compute,
20% price). Closed models are labelled **confidence: low**.

### c) Device model — local models (Ollama)
```
Wh/1k = device_watts / tokens_per_second × 1000 / 3600
```
Set `device_watts` and `tps` in the catalog from your own measurement (e.g. `powermetrics` on macOS,
`nvidia-smi` on Linux). A notable result: a laptop running a 3B model at 25 W / 55 tok/s uses
**more** energy per token than a large data centre serving a model of similar size, because the data
centre batches many users onto each GPU. Local inference still wins on cost (zero), on-site water
(zero), privacy and offline use — and can be forced with `require_local`.

### Prefill
Input tokens are processed in parallel and are compute-bound, so they cost far less per token than
generated tokens. They are charged at **10%** of an output token.

### Efficiency labels
Like an EU appliance label, on Wh per 1k output tokens: A < 0.02, B < 0.05, C < 0.1, D < 0.25,
E < 0.5, F < 1.0, G ≥ 1.0.

## 3. Carbon and water

```
carbon_g = energy_kWh × grid_intensity_gCO2e_per_kWh(region)
water_mL = (energy_kWh / PUE × WUE_on-site + energy_kWh × water_intensity_off-site(region)) × 1000
```

* **Grid intensity** comes from `data/grid.yaml` (approximate annual averages) or, if
  `ELECTRICITYMAPS_API_KEY` is set, from the live Electricity Maps API every 15 minutes — so routing
  decisions become carbon-aware in real time.
* **On-site water** (WUE) is evaporative cooling at the data centre, per provider in the catalog.
  It applies to IT energy, hence the division by PUE. Local devices have WUE = 0.
* **Off-site water** is water consumed generating the electricity (thermal plant cooling,
  reservoir evaporation), applied to total energy.

## 4. Scores and ranking (`ecoroute/router.py`)

For each candidate, the request's estimated impact is turned into 0–100 scores on fixed log scales
(so scores are comparable across requests):

* **Sustainability** = 50% energy + 30% carbon + 20% water
  (bounds: 0.003–3 Wh, 0.001–2 g, 0.01–15 mL)
* **Cost** (bounds: $0.000005–$0.05; free = 100)
* **Quality fit**: `min(1, quality / (40 + 12 × tier))`, blended 70/30 with absolute quality.
  A model far above what the tier needs gets **no extra credit** — that headroom is wasted energy.
* **Latency** from tokens/second (10 → 0, 1000 → 100)

Policy weights (sustainability / cost / quality / latency):

| policy | sust. | cost | quality | latency |
|---|---|---|---|---|
| eco | 0.60 | 0.20 | 0.15 | 0.05 |
| balanced | 0.35 | 0.25 | 0.30 | 0.10 |
| quality | 0.15 | 0.10 | 0.65 | 0.10 |
| cost | 0.20 | 0.65 | 0.10 | 0.05 |

The quality weight is further multiplied by `0.6 + 0.8 × complexity`, so quality matters less for a
greeting than for a proof. Models whose `max_tier` is below the prompt's tier are excluded outright.

**Budgets.** If month-to-date spend reaches the USD budget the policy switches to *cost*; if energy
reaches the Wh budget it switches to *eco*; above 80% of spend, *quality* softens to *balanced*.

**Baseline.** Every request is also costed as if it had been sent to the baseline model
(`ECOROUTE_BASELINE_MODEL`, default GPT-5) with the same token counts. All "saved" figures in the
dashboard are relative to that counterfactual.

## 5. Caching

Responses are cached by a hash of the normalised conversation and the required tier. A cache hit
records zero energy, carbon, water and cost — the cheapest inference is the one you don't run.

## 6. Limitations

* All energy figures are **estimates**. Providers rarely disclose per-request energy; hardware,
  batch size, utilisation, speculative decoding and hidden reasoning tokens all move the true number.
  Reasoning tokens *are* counted when the provider reports them in usage.
* Groq runs on LPUs rather than GPUs; the GPU-based compute model is an approximation for it.
* Price is a noisy proxy: discounts, loss-leading prices and different margins break the link.
* The complexity classifier is a heuristic. A learned classifier trained on a larger labelled set
  (or a tiny LLM judge) would be a natural next step.
* Quality scores in the catalog are relative judgements, not benchmark results.

## Useful references

* Luccioni, Jernite & Strubell, *Power Hungry Processing: Watts Driving the Cost of AI Deployment?* (2024)
* Li, Yang, Islam & Ren, *Making AI Less "Thirsty": Uncovering and Addressing the Secret Water Footprint of AI Models* (2023)
* Patterson et al., *Carbon Emissions and Large Neural Network Training* (2021)
* Ong et al., *RouteLLM: Learning to Route LLMs with Preference Data* (2024)
* Chen, Zaharia & Zou, *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance* (2023)
* Google, *Measuring the environmental impact of delivering AI at Google scale* (2025)
* Epoch AI, *How much energy does ChatGPT use?* (2025)
* Electricity Maps API — https://www.electricitymaps.com
