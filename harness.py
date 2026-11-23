"""Pure, testable prompt/scoring helpers. No model or hidden data is loaded here."""
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re

INSTRUCTION = (
    "Solve the following math problem. Reason step by step, "
    "then give the final numeric answer on a new line in the format '#### <answer>'."
)
NUMBER = r"[+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def clean_solution(text):
    return re.sub(r"<<[^>]*>>", "", text).strip()


def canonical(text):
    try:
        value = Decimal(str(text).replace(",", "").strip())
        if not value.is_finite():
            return None
        return "0" if value == 0 else format(value.normalize(), "f")
    except (InvalidOperation, ValueError):
        return None


def parse_answer(text):
    """Prefer the last #### answer; otherwise use the last number, as in tutorial.

    Compare Decimal values exactly, without the tutorial's rounding to 2 decimals.
    The same parser is fixed for every round. Missing answers are always incorrect.
    """
    marked = re.findall(r"####\s*(" + NUMBER + r")", text)
    numbers = marked or re.findall(NUMBER, text)
    return canonical(numbers[-1]) if numbers else None


def gold_answer(text):
    if "####" not in text:
        raise ValueError("Malformed gold answer")
    result = canonical(text.rsplit("####", 1)[-1])
    if result is None:
        raise ValueError("Non-numeric gold answer")
    return result


def has_marker(text):
    return bool(re.search(r"####\s*(" + NUMBER + r")", text))


def vote(responses):
    answers = [parse_answer(x) for x in responses]
    counts = Counter(x for x in answers if x is not None)
    # Counter preserves first appearance: a tied greedy answer wins if valid.
    return counts.most_common(1)[0][0] if counts else None


def needs_more(responses):
    counts = Counter(parse_answer(x) for x in responses)
    counts.pop(None, None)
    return not counts or max(counts.values()) < 2


class Retriever:
    """Small TF-IDF retriever over ONLY the visible demonstration questions."""
    STOP = set("a an the is are was were of to in for and on at with how many much does do".split())

    def __init__(self, demos):
        self.demos = demos
        documents = [self.tokens(x["question"]) for x in demos]
        df = Counter(t for doc in documents for t in set(doc))
        self.idf = {t: math.log((1 + len(demos)) / (1 + n)) + 1 for t, n in df.items()}
        self.vectors = [self.vector(doc) for doc in documents]

    def tokens(self, text):
        return [t for t in re.findall(r"[a-z]+|\d+", text.lower())
                if t not in self.STOP and not t.isdigit()]

    def vector(self, tokens):
        v = {t: n * self.idf.get(t, 0) for t, n in Counter(tokens).items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1
        return {t: x / norm for t, x in v.items()}

    def select(self, question, k):
        q = self.vector(self.tokens(question))
        scores = [sum(q.get(t, 0) * x for t, x in v.items()) for v in self.vectors]
        return sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]


def validate_config(config, pool_size):
    instruction = config.get("instruction")
    ids = config.get("demo_ids", [])
    if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 1200:
        raise ValueError("instruction must be 1-1200 characters")
    if not isinstance(ids, list) or len(ids) > 3 or len(ids) != len(set(ids)):
        raise ValueError("demo_ids must contain at most three distinct IDs")
    if any(type(i) is not int or not 0 <= i < pool_size for i in ids):
        raise ValueError("Invalid demo index")
    return {"instruction": instruction.strip(), "demo_ids": ids}


def build_prompt(question, config, demos, retriever):
    ids = (retriever.select(question, config.get("k", 2))
           if config.get("retrieval") else config["demo_ids"])
    parts = [config["instruction"]]
    for i in ids:
        ex = demos[i]
        parts.append(f"Problem: {ex['question']}\nSolution: {clean_solution(ex['answer'])}")
    parts.append(f"Problem: {question}\nSolution:")
    return "\n\n".join(parts), ids
