import argparse
import json
import re
import time
from collections import Counter
from typing import Optional
import random

from datasets import load_dataset
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

ARC_SYSTEM = (
    "You are an expert at multiple-choice science questions. "
    "Think step by step, then state your final answer as a single letter "
    "on its own line in the format: Answer: X"
)

GSM8K_SYSTEM = (
    "You are an expert math tutor. Solve the problem step by step. "
    "At the end, write your final numeric answer on its own line in the format: Answer: <number>"
)


def build_arc_prompt(item: dict) -> tuple[str, str]:
    choices = item["choices"]
    lines = [f"Question: {item['question']}", ""]
    for label, text in zip(choices["label"], choices["text"]):
        lines.append(f"  {label}) {text}")
    lines += ["", "Think step by step, then give your answer."]
    gold = item["answerKey"].upper()
    return "\n".join(lines), gold


def build_gsm8k_prompt(item: dict) -> tuple[str, str]:
    prompt = f"Problem: {item['question']}\n\nSolve step by step."
    # Extract numeric answer from GSM8K answer field (after ####)
    raw_answer = item["answer"]
    match = re.search(r"####\s*([\d,\.\-]+)", raw_answer)
    gold = match.group(1).replace(",", "").strip() if match else raw_answer.strip()
    return prompt, gold


# ─────────────────────────────────────────────────────────────────────────────
# ANSWER EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_arc_answer(text: str, valid_labels: list) -> Optional[str]:
    match = re.search(r"Answer\s*:\s*([A-Da-d])", text)
    if match:
        ans = match.group(1).upper()
        if ans in [l.upper() for l in valid_labels]:
            return ans
    letters = re.findall(r"\b([A-D])\b", text.upper())
    for letter in reversed(letters):
        if letter in [l.upper() for l in valid_labels]:
            return letter
    return None


def extract_gsm8k_answer(text: str) -> Optional[str]:
    match = re.search(r"Answer\s*:\s*([\d,\.\-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    # fallback: last number in text
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else None


def answers_match_gsm8k(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-3
    except ValueError:
        return pred.strip() == gold.strip()


# ─────────────────────────────────────────────────────────────────────────────
# BACKENDS
# ─────────────────────────────────────────────────────────────────────────────

class OllamaBackend:
    def __init__(self, model: str, temperature: float, max_tokens: int,
                 base_url: str = "http://localhost:11434"):
        import requests
        self._req = requests
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.url = base_url.rstrip("/") + "/api/chat"

    def generate(self, system: str, prompt: str, n: int) -> list[str]:
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": prompt}]
        outputs = []
        for _ in range(n):
            r = self._req.post(self.url, json={
                "model": self.model, "messages": msgs, "stream": False,
                "options": {"temperature": self.temperature, "num_predict": self.max_tokens}
            }, timeout=180)
            r.raise_for_status()
            outputs.append(r.json()["message"]["content"])
        return outputs


class HuggingFaceBackend:
    def __init__(self, model: str, temperature: float, max_tokens: int, quantize: str = "none"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

        print(f"  Loading {model} (quantize={quantize}) …")
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        quant_config = None
        if quantize == "4bit":
            quant_config = BitsAndBytesConfig(load_in_4bit=True,
                                              bnb_4bit_compute_dtype=torch.bfloat16)
        elif quantize == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        mdl = AutoModelForCausalLM.from_pretrained(
            model, quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if quant_config is None else None,
            device_map="auto", trust_remote_code=True,
        )
        self.pipe = pipeline("text-generation", model=mdl, tokenizer=tokenizer)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system: str, prompt: str, n: int) -> list[str]:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        outputs = []
        for _ in range(n):
            result = self.pipe(msgs, max_new_tokens=self.max_tokens,
                               temperature=self.temperature, do_sample=True)
            text = result[0]["generated_text"]
            if isinstance(text, list):
                text = text[-1].get("content", "")
            outputs.append(text)
        return outputs


class OpenAIBackend:
    def __init__(self, model: str, temperature: float, max_tokens: int):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system: str, prompt: str, n: int) -> list[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            n=n, temperature=self.temperature, max_tokens=self.max_tokens,
        )
        return [c.message.content for c in response.choices]


# ─────────────────────────────────────────────────────────────────────────────
# SELF-VERIFICATION (scoring + re-ranking)
# ─────────────────────────────────────────────────────────────────────────────

VERIFY_SYSTEM = (
    "You are a strict answer verifier. Given a problem and a candidate solution, "
    "rate the solution's correctness on a scale of 1-10 where 10=definitely correct. "
    "Respond ONLY with a JSON object: {\"score\": <number>, \"reason\": \"<one sentence>\"}"
)

def self_verify(backend, system: str, prompt: str, solution: str) -> float:
    """Ask the model to score its own solution. Returns float 1-10."""
    verify_prompt = (
        f"Original problem:\n{prompt}\n\n"
        f"Candidate solution:\n{solution}\n\n"
        "Rate this solution's correctness (1-10)."
    )
    try:
        raw = backend.generate(VERIFY_SYSTEM, verify_prompt, n=1)[0]
        match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', raw)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return 5.0  # neutral fallback


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATION STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

def majority_vote(answers: list) -> Optional[str]:
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def best_by_verification(backend, prompt: str, system: str,
                          outputs: list[str], answers: list) -> Optional[str]:
    """Score each output and return the answer with the highest score."""
    if not any(a is not None for a in answers):
        return None
    scored = []
    for out, ans in zip(outputs, answers):
        if ans is not None:
            score = self_verify(backend, system, prompt, out)
            scored.append((score, ans))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXPERIMENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(args):
    # ── Dataset ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" Loading dataset: {args.dataset.upper()}")
    print(f"{'='*60}")

    if args.dataset == "arc":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=args.split)
        build_prompt = build_arc_prompt
        extract_fn = lambda text, item: extract_arc_answer(
            text, item["choices"]["label"])
        is_correct = lambda pred, gold: (pred or "").upper() == gold.upper()
        system_prompt = ARC_SYSTEM
    elif args.dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=args.split)
        build_prompt = build_gsm8k_prompt
        extract_fn = lambda text, item: extract_gsm8k_answer(text)
        is_correct = lambda pred, gold: answers_match_gsm8k(pred, gold)
        system_prompt = GSM8K_SYSTEM
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.max_questions:
        indices = list(range(len(ds)))
        random.seed(42)
        random.shuffle(indices)
        ds = ds.select(indices[:args.max_questions])

    print(f"  Questions: {len(ds)}")

    # ── Backends ─────────────────────────────────────────────────────────────
    def make_backend(model, backend_type, quantize="none"):
        if backend_type == "ollama":
            return OllamaBackend(model, args.temperature, args.max_tokens)
        elif backend_type == "hf":
            return HuggingFaceBackend(model, args.temperature, args.max_tokens, quantize)
        elif backend_type == "openai":
            return OpenAIBackend(model, args.temperature, args.max_tokens)
        raise ValueError(f"Unknown backend: {backend_type}")

    max_n = max(args.n_values)

    print(f"\n  Student model : {args.student_model} (backend={args.backend})")
    print(f"  Teacher model : {args.teacher_model} (backend={args.teacher_backend})")
    print(f"  N values      : {args.n_values}")
    print(f"  Aggregation   : {args.aggregation}\n")

    student_backend = make_backend(args.student_model, args.backend, args.quantize)
    teacher_backend = make_backend(args.teacher_model, args.teacher_backend)

    # ── Result accumulators ─────────────────────────────────────────────────
    # n_results[n] = list of {correct, prediction, gold, ...}
    n_results   = {n: [] for n in args.n_values}
    teacher_results = []
    per_question_data = []

    # ── Evaluation loop ──────────────────────────────────────────────────────
    for item in tqdm(ds, desc="Evaluating"):
        prompt, gold = build_prompt(item)

        # ── Student: generate max_n samples once, slice for each N ──────────
        t0 = time.time()
        student_outputs = student_backend.generate(system_prompt, prompt, max_n)
        student_gen_time = time.time() - t0

        student_answers = [extract_fn(out, item) for out in student_outputs]

        q_record = {
            "question":    prompt[:200],
            "gold":        gold,
            "student_raw": student_outputs,
            "student_ans": student_answers,
            "n_predictions": {},
            "gen_time_student": round(student_gen_time, 2),
        }

        for n in args.n_values:
            subset_outputs  = student_outputs[:n]
            subset_answers  = student_answers[:n]

            if args.aggregation == "majority":
                pred = majority_vote(subset_answers)
            elif args.aggregation == "verification":
                pred = best_by_verification(
                    student_backend, prompt, system_prompt,
                    subset_outputs, subset_answers)
            else:
                pred = majority_vote(subset_answers)

            correct = is_correct(pred, gold)
            n_results[n].append({"correct": correct, "prediction": pred, "gold": gold})
            q_record["n_predictions"][str(n)] = {"pred": pred, "correct": correct}

        # ── Teacher: single sample ────────────────────────────────────────────
        t1 = time.time()
        teacher_out = teacher_backend.generate(system_prompt, prompt, 1)[0]
        teacher_time = time.time() - t1

        teacher_pred = extract_fn(teacher_out, item)
        t_correct = is_correct(teacher_pred, gold)
        teacher_results.append({"correct": t_correct, "prediction": teacher_pred, "gold": gold})

        q_record["teacher_pred"]    = teacher_pred
        q_record["teacher_correct"] = t_correct
        q_record["gen_time_teacher"] = round(teacher_time, 2)
        per_question_data.append(q_record)

        # Live log
        student_n1_pred = q_record["n_predictions"].get("1", {}).get("pred", "?")
        tqdm.write(
            f"  gold={gold}  student_n1={student_n1_pred}  "
            f"teacher={teacher_pred}  ({'✓' if t_correct else '✗'}teacher)"
        )

    # ── Aggregate metrics ────────────────────────────────────────────────────
    student_summary = {}
    for n in args.n_values:
        records = n_results[n]
        acc = sum(r["correct"] for r in records) / len(records)
        student_summary[n] = {"accuracy": round(acc, 4), "n_questions": len(records)}

    t_acc = sum(r["correct"] for r in teacher_results) / len(teacher_results)
    teacher_summary = {"accuracy": round(t_acc, 4), "n_questions": len(teacher_results)}

    # Compute crossover: smallest N where student >= teacher accuracy
    crossover_n = None
    for n in sorted(args.n_values):
        if student_summary[n]["accuracy"] >= teacher_summary["accuracy"]:
            crossover_n = n
            break

    output_data = {
        "experiment": {
            "student_model":   args.student_model,
            "teacher_model":   args.teacher_model,
            "backend":         args.backend,
            "teacher_backend": args.teacher_backend,
            "quantize":        args.quantize,
            "dataset":         args.dataset,
            "split":           args.split,
            "n_values":        args.n_values,
            "aggregation":     args.aggregation,
            "temperature":     args.temperature,
            "n_questions":     len(ds),
        },
        "student_results": student_summary,
        "teacher_results": teacher_summary,
        "crossover_n":     crossover_n,
        "per_question":    per_question_data,
    }

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(" RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Teacher ({args.teacher_model}) N=1: {teacher_summary['accuracy']:.1%}")
    for n in args.n_values:
        s = student_summary[n]
        delta = s["accuracy"] - teacher_summary["accuracy"]
        marker = " ← CROSSOVER" if n == crossover_n else ""
        print(f"  Student ({args.student_model}) N={n:>2}: "
              f"{s['accuracy']:.1%}  (Δ{delta:+.1%}){marker}")
    if crossover_n:
        print(f"\n  ✓ Crossover achieved at N={crossover_n}")
    else:
        print(f"\n  ✗ No crossover achieved within tested N values")
    print(f"{'='*60}\n")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = args.output or "results.json"
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to: {out_path}")
    return output_data


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student_model",   default="qwen2.5:3b")
    p.add_argument("--teacher_model",   default="qwen2.5:7b")
    p.add_argument("--backend",         default="ollama",
                   choices=["ollama", "hf", "openai"])
    p.add_argument("--teacher_backend", default="ollama",
                   choices=["ollama", "hf", "openai"])
    p.add_argument("--quantize",        default="4bit",
                   choices=["none", "4bit", "8bit"],
                   help="Quantization for HF backend only")
    p.add_argument("--dataset",         default="arc",
                   choices=["arc", "gsm8k"])
    p.add_argument("--split",           default="test",
                   choices=["train", "validation", "test"])
    p.add_argument("--n_values",        nargs="+", type=int,
                   default=[1, 3, 5, 10],
                   help="List of N values to test (Best-of-N)")
    p.add_argument("--aggregation",     default="majority",
                   choices=["majority", "verification"],
                   help="majority=majority vote, verification=self-verification re-ranking")
    p.add_argument("--temperature",     type=float, default=0.7)
    p.add_argument("--max_tokens",      type=int,   default=512)
    p.add_argument("--max_questions",   type=int,   default=50)
    p.add_argument("--output",          default="results.json")
    return p.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
