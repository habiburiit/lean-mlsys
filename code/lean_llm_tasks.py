#!/usr/bin/env python
"""
lean_llm_tasks.py  -  self-contained 0-shot multiple-choice evaluator for the tasks AttentionBreaker reports through the
LM-Evaluation-Harness (ARC-Easy, ARC-Challenge, BoolQ, HellaSwag, PIQA, OpenBookQA, SocialIQA), for servers without pip.
Protocol as in the harness: each choice is scored by the log-likelihood of its continuation given the prompt; ARC, HellaSwag,
PIQA and OpenBookQA report acc_norm (log-likelihood divided by continuation length), BoolQ and SocialIQA report acc.
Datasets come from Hugging Face (same route as WikiText / MMLU in the other scripts).

  from lean_llm_tasks import evaluate;  evaluate(model, tok, dev, ["arc_easy", "boolq"], limit=None) -> {task: accuracy %}
"""
import torch, torch.nn.functional as F
from datasets import load_dataset


def _arc(cfg, limit):
    ds = load_dataset("allenai/ai2_arc", cfg, split="test")
    out = []
    for r in ds:
        labels = r["choices"]["label"]; texts = r["choices"]["text"]
        if r["answerKey"] not in labels: continue
        out.append((f"Question: {r['question']}\nAnswer:", [" " + t for t in texts], labels.index(r["answerKey"]), True))
    return out[:limit] if limit else out


def _boolq(limit):
    ds = load_dataset("google/boolq", split="validation")
    out = [(f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:", [" no", " yes"], int(bool(r["answer"])), False) for r in ds]
    return out[:limit] if limit else out


def _hellaswag(limit):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    out = []
    for r in ds:
        ctx = (r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize()).strip()
        out.append((ctx, [" " + e for e in r["endings"]], int(r["label"]), True))
    return out[:limit] if limit else out


def _piqa(limit):
    ds = None
    for args, kw in [(("ybisk/piqa",), {"revision": "refs/convert/parquet"}), (("ybisk/piqa",), {}), (("lighteval/piqa_helm",), {})]:
        try:
            ds = load_dataset(*args, split="validation", **kw); break
        except Exception as e:
            print(f"   [piqa] {args} {kw}: {str(e)[:100]}", flush=True)
    if ds is None: raise RuntimeError("PIQA not loadable on this datasets version")
    out = [(f"Question: {r['goal']}\nAnswer:", [" " + r["sol1"], " " + r["sol2"]], int(r["label"]), True) for r in ds]
    return out[:limit] if limit else out


def _openbookqa(limit):
    ds = load_dataset("allenai/openbookqa", "main", split="test")
    out = []
    for r in ds:
        labels = r["choices"]["label"]; texts = r["choices"]["text"]
        out.append((r["question_stem"], [" " + t for t in texts], labels.index(r["answerKey"]), True))
    return out[:limit] if limit else out


def _siqa(limit):
    ds = load_dataset("allenai/social_i_qa", split="validation")
    out = [(f"Q: {r['context']} {r['question']}\nA:", [" " + r["answerA"], " " + r["answerB"], " " + r["answerC"]], int(r["label"]) - 1, False) for r in ds]
    return out[:limit] if limit else out


LOADERS = {"arc_easy": lambda l: _arc("ARC-Easy", l), "arc_challenge": lambda l: _arc("ARC-Challenge", l), "boolq": _boolq,
           "hellaswag": _hellaswag, "piqa": _piqa, "openbookqa": _openbookqa, "social_iqa": _siqa}


@torch.no_grad()
def _choice_loglik(model, tok, dev, prompt, choices, max_len=1024):
    """sum log p(continuation | prompt) for every choice, and the continuation length in characters"""
    ctx = tok(prompt, add_special_tokens=False)["input_ids"]
    seqs, nconts = [], []
    for c in choices:
        full = tok(prompt + c, add_special_tokens=False)["input_ids"]
        n = len(full) - len(ctx)
        if n <= 0: full = ctx + tok(c, add_special_tokens=False)["input_ids"]; n = len(full) - len(ctx)
        full = full[-max_len:]; seqs.append(full); nconts.append(n)
    L = max(len(s) for s in seqs); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(seqs), L), pad, dtype=torch.long); att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs): ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
    logits = model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits.float()
    lp = F.log_softmax(logits[:, :-1], -1); tgt = ids[:, 1:].to(dev); tl = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    out = []
    for i, s in enumerate(seqs):
        n = nconts[i]; e = len(s) - 1; out.append(float(tl[i, e - n:e].sum()))
    return out, [len(c) for c in choices]


def evaluate(model, tok, dev, tasks, limit=None, log=print):
    res = {}
    for t in tasks:
        try:
            data = LOADERS[t](limit)
        except Exception as e:
            if log: log(f"   [task] {t}: SKIPPED ({str(e)[:120]})")
            res[t] = None; continue
        correct = 0
        for k, (prompt, choices, label, norm) in enumerate(data):
            ll, ln = _choice_loglik(model, tok, dev, prompt, choices)
            score = [a / b for a, b in zip(ll, ln)] if norm else ll
            correct += int(max(range(len(score)), key=lambda j: score[j]) == label)
            if log and (k + 1) % 1000 == 0: log(f"      {t} {k+1}/{len(data)}  acc so far {100*correct/(k+1):.1f}")
        res[t] = 100.0 * correct / max(len(data), 1)
        if log: log(f"   [task] {t}: {res[t]:.2f}  ({len(data)} examples, {'acc_norm' if data and data[0][3] else 'acc'})")
    return res
