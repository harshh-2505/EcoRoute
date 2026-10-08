"""Prompt complexity analysis.

A transparent, dependency-free scorer: it extracts interpretable features from the prompt,
combines them into a 0-1 score and maps that to one of five tiers. Every contribution is
returned so the dashboard can explain *why* a prompt was judged simple or hard.

Tiers
  1 trivial   greetings, one-line facts, tiny rewrites
  2 simple    short explanations, summaries, translations, formatting
  3 moderate  multi-paragraph writing, standard code, comparisons
  4 complex   multi-step reasoning, non-trivial code/maths, structured analysis
  5 expert    research-grade, proofs, system design, long multi-constraint tasks
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

TIER_NAMES = {1: "trivial", 2: "simple", 3: "moderate", 4: "complex", 5: "expert"}
TIER_THRESHOLDS = (0.12, 0.20, 0.48, 0.74)  # calibrated on benchmarks/prompts.jsonl (train split)  # score cut points between tiers
EXPECTED_OUTPUT_TOKENS = {1: 60, 2: 220, 3: 500, 4: 900, 5: 1600}


def _rx(*words: str) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)


TRIVIAL = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|ok|okay|good (morning|night|evening)|bye|yo|sup)\b[\s!.?]*$",
    re.IGNORECASE)
REASONING = _rx(r"why", r"explain(?: why| how)?", r"analy[sz]e", r"analysis", r"compare", r"contrast",
                r"evaluate", r"assess", r"critique", r"justify", r"reason(?:ing)?", r"implications?",
                r"trade-?offs?", r"pros and cons", r"step[- ]by[- ]step", r"argue", r"infer")
DEEP_REASONING = _rx(r"prove", r"proof", r"derive", r"derivation", r"formal(?:ly)?", r"rigorous(?:ly)?",
                     r"theorem", r"lemma", r"first principles", r"counter-?example", r"optimal(?:ity)?",
                     r"complexity analysis", r"edge cases")
CODE = _rx(r"code", r"function", r"class", r"script", r"implement(?:ation)?", r"debug", r"refactor",
           r"algorithm", r"api", r"sql", r"regex", r"python", r"javascript", r"typescript", r"java",
           r"c\+\+", r"rust", r"golang", r"bug", r"compile", r"unit tests?", r"endpoint", r"database",
           r"react", r"docker", r"kubernetes")
MATH = _rx(r"integral", r"derivative", r"equation", r"probability", r"matrix", r"eigen\w*", r"solve",
           r"calculate", r"compute", r"statistic(?:s|al)?", r"variance", r"optimi[sz]e", r"gradient",
           r"limit", r"series", r"polynomial", r"differential", r"primes?", r"number theory",
           r"algebra", r"geometry", r"theorem")
SYSTEM_DESIGN = _rx(r"architect(?:ure)?", r"system design", r"scalab(?:le|ility)", r"distributed",
                    r"microservices?", r"design (?:a|an|the) system", r"infrastructure", r"pipeline",
                    r"end[- ]to[- ]end", r"production[- ]ready", r"fault[- ]toleran\w+")
EXPERT_DOMAIN = _rx(r"legal", r"contract", r"liability", r"diagnos\w+", r"clinical", r"pharmacolog\w+",
                    r"regulat\w+", r"compliance", r"financial model", r"valuation", r"research paper",
                    r"literature review", r"hypothes[ie]s", r"peer[- ]review", r"thesis", r"quantum",
                    r"cryptograph\w+", r"security audit", r"vulnerabilit\w+")
LONG_FORM = _rx(r"essay", r"report", r"article", r"blog post", r"story", r"chapter", r"whitepaper",
                r"proposal", r"documentation", r"tutorial", r"guide", r"lesson plan", r"business plan",
                r"cover letter", r"speech")
FACTOID = _rx(r"translate", r"what is(?! the difference)", r"who is", r"who was", r"who wrote", r"when (?:is|was|did)",
              r"where is", r"capital of", r"convert", r"synonyms?", r"antonyms?", r"define", r"definition",
              r"spell(?:ing)?", r"one word", r"yes or no", r"tagline", r"title for", r"how many", r"how old")
GENERATIVE = _rx(r"write", r"create", r"draft", r"describe", r"summari[sz]e", r"list", r"plan", r"generate",
                 r"outline", r"email", r"rewrite", r"fix", r"difference between", r"how (?:does|do|to)",
                 r"tips", r"suggest(?:ions)?", r"ideas", r"examples?")
DEPTH = _rx(r"detailed", r"comprehensive", r"in[- ]depth", r"thorough(?:ly)?", r"exhaustive", r"each",
            r"every", r"covering", r"including", r"considering", r"with references", r"justify")
BRIEF = _rx(r"short", r"briefly", r"brief", r"in one sentence", r"concise(?:ly)?", r"quick(?:ly)?",
            r"simple terms", r"one example")
CONSTRAINT = _rx(r"must", r"ensure", r"exactly", r"at least", r"at most", r"no more than", r"format(?:ted)? as",
                 r"json", r"yaml", r"table", r"markdown", r"bullet points", r"include", r"without using",
                 r"requirements?", r"constraints?")
WORD_TARGET = re.compile(r"(\d{2,5})\s*(?:-|to)?\s*(?:\d{2,5}\s*)?(words|word|pages|page|tokens|lines)",
                         re.IGNORECASE)
CODE_BLOCK = re.compile(r"```|^\s{4,}\S|[{};]\s*$|def \w+\(|function \w+\(|=>", re.MULTILINE)
MATH_SYMBOLS = re.compile(r"[∫∑∏√≤≥≠±∞]|\\(?:frac|int|sum|sqrt)|\b\d+\s*[\^*/]\s*\d+|\bx\^\d")
ENUMERATION = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+", re.MULTILINE)


@dataclass
class ComplexityResult:
    score: float
    tier: int
    tier_name: str
    task_type: str
    expected_output_tokens: int
    input_tokens: int
    features: dict = field(default_factory=dict)
    contributions: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_tokens(text: str) -> int:
    """~4 chars/token for English, but never fewer than words."""
    if not text:
        return 0
    return max(len(text.split()), round(len(text) / 4))


def _hits(rx: re.Pattern, text: str) -> int:
    return len(rx.findall(text))


def _sat(n: float, k: float) -> float:
    """Saturating 0..1 transform: 1 hit -> ~k/(k+1)... smooth diminishing returns."""
    return n / (n + k) if n > 0 else 0.0


def analyze(prompt: str, history_tokens: int = 0) -> ComplexityResult:
    text = (prompt or "").strip()
    tokens = estimate_tokens(text)
    lower = text.lower()

    f = {
        "tokens": tokens,
        "reasoning": _hits(REASONING, lower),
        "deep_reasoning": _hits(DEEP_REASONING, lower),
        "code": _hits(CODE, lower) + 2 * len(CODE_BLOCK.findall(text)),
        "math": _hits(MATH, lower) + 2 * len(MATH_SYMBOLS.findall(text)),
        "system_design": _hits(SYSTEM_DESIGN, lower),
        "expert_domain": _hits(EXPERT_DOMAIN, lower),
        "long_form": _hits(LONG_FORM, lower),
        "factoid": _hits(FACTOID, lower),
        "generative": _hits(GENERATIVE, lower),
        "depth": _hits(DEPTH, lower),
        "brief": _hits(BRIEF, lower),
        "clauses": lower.count(",") + lower.count(" and ") + lower.count(";"),
        "constraints": _hits(CONSTRAINT, lower),
        "questions": text.count("?"),
        "enumerated_items": len(ENUMERATION.findall(text)),
        "trivial": bool(TRIVIAL.match(text)) or tokens <= 3,
    }
    word_target = 0
    for m in WORD_TARGET.finditer(lower):
        n, unit = int(m.group(1)), m.group(2)
        n = n * 500 if unit.startswith("page") else n * 12 if unit.startswith("line") else n
        word_target = max(word_target, n)
    f["requested_length_words"] = word_target

    c: dict[str, float] = {}
    c["length"] = 0.25 * _sat(max(tokens - 8, 0), 80)
    c["reasoning"] = 0.30 * _sat(f["reasoning"], 1.0)
    c["deep_reasoning"] = 0.35 * _sat(f["deep_reasoning"], 0.8)
    c["code"] = 0.35 * _sat(f["code"], 1.0)
    c["math"] = 0.25 * _sat(f["math"], 1.5)
    c["system_design"] = 0.40 * _sat(f["system_design"], 1.2)
    c["expert_domain"] = 0.25 * _sat(f["expert_domain"], 1.0)
    c["long_form"] = 0.25 * _sat(f["long_form"], 1.0)
    c["constraints"] = 0.12 * _sat(f["constraints"], 2.0)
    c["multi_part"] = 0.15 * _sat(max(f["questions"] - 1, 0) + f["enumerated_items"], 3.0)
    c["clauses"] = 0.15 * _sat(max(f["clauses"] - 1, 0), 3.0)
    c["generative"] = 0.18 * _sat(f["generative"], 1.0)
    c["depth"] = 0.15 * _sat(f["depth"], 1.5)
    c["requested_length"] = 0.35 * _sat(word_target, 400)
    c["history"] = 0.08 * _sat(history_tokens, 2000)
    # Simple-task cues pull the score down, but less so when hard signals are present.
    hard = c["deep_reasoning"] + c["code"] + c["math"] + c["system_design"]
    softener = 1 - min(hard / 0.3, 1)
    c["factoid"] = -0.20 * _sat(f["factoid"], 1.0) * softener
    c["brief"] = -0.10 * _sat(f["brief"], 1.0) * softener
    if f["trivial"]:
        c["trivial"] = -0.5

    score = max(0.0, min(1.0, 0.04 + sum(c.values())))
    tier = 1 + sum(score >= t for t in TIER_THRESHOLDS)

    # Hard floors: some signals shouldn't be sent to tiny models regardless of length.
    floor = 1
    if f["code"] >= 1 or f["math"] >= 2:
        floor = 2
    if f["deep_reasoning"] >= 1 or f["code"] >= 4:
        floor = 3
    if f["deep_reasoning"] >= 3 or f["system_design"] >= 3:
        floor = 4
    tier = max(tier, floor)

    expected = EXPECTED_OUTPUT_TOKENS[tier]
    if word_target:
        expected = max(expected, int(word_target * 1.35))
    task_type = _task_type(f)
    return ComplexityResult(
        score=round(score, 4), tier=tier, tier_name=TIER_NAMES[tier], task_type=task_type,
        expected_output_tokens=expected, input_tokens=tokens + history_tokens,
        features=f, contributions={k: round(v, 4) for k, v in c.items() if v},
    )


def _task_type(f: dict) -> str:
    if f["trivial"]:
        return "chit-chat"
    ranked = sorted(
        [("system design", f["system_design"] * 1.5), ("coding", f["code"]), ("math", f["math"]),
         ("reasoning", f["reasoning"] + 1.5 * f["deep_reasoning"]), ("expert domain", f["expert_domain"]),
         ("long-form writing", f["long_form"] * 1.2), ("quick task", f["factoid"] * 0.9), ("writing", f["generative"] * 0.5)],
        key=lambda kv: kv[1], reverse=True)
    return ranked[0][0] if ranked[0][1] > 0 else "general"
