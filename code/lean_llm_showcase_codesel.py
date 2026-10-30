#!/usr/bin/env python
"""
lean_llm_showcase_codesel.py  -  lean_llm_showcase.py with a different CODE PLACEMENT rule (defender side, no retraining):
  --code-score gradw     : |gradient| x |weight|            (the original rule in lean_llm_showcase.py)
  --code-score attacker  : the attacker's own per-weight score (|gradient| x MSB step of the deployed quantiser), so the ECC
                           covers exactly the weights whose single flip hurts most in the format that is attacked.
Why: under per-channel INT8 the two rankings differ, and a LEAN model (which concentrates its damage in a small tail) is only
protected if the code sits on that tail.  lean_llm_showcase.py is NOT modified: its source is loaded and one line is replaced.
usage: same as lean_llm_showcase.py plus --code-score; e.g.
  python -u lean_llm_bf16.py lean_llm_showcase_codesel.py --model ckpt_llm/qwen25_7b/lam10_sum --cov 0.005 --seed 0 --per-channel \
      --code-score attacker --tag qwen25_7b_lam10_sum_pcsel --out results_llm/showcase
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "lean_llm_showcase.py")).read()
old = "w = E.dequant_any(clean[n], layers[n][1], layers[n][2]).reshape(-1); dmg[n] = np.abs(np.asarray(ab.grads[n]).reshape(-1)) * np.abs(w)"
new = ("w = E.dequant_any(clean[n], layers[n][1], layers[n][2]).reshape(-1)\n"
       "                    if a.code_score == 'attacker' and n in ab.scores:\n"
       "                        sc0 = np.asarray(ab.scores[n]).reshape(-1).astype(np.float64); sc0[~np.isfinite(sc0)] = 0.0; dmg[n] = np.abs(sc0)\n"
       "                    else: dmg[n] = np.abs(np.asarray(ab.grads[n]).reshape(-1)) * np.abs(w)")
assert old in src, "lean_llm_showcase.py changed; the placement line was not found"
src = src.replace(old, new)
src = src.replace('ap.add_argument("--fixed", type=int, default=0,', 'ap.add_argument("--code-score", choices=["gradw", "attacker"], default="attacker"); ap.add_argument("--fixed", type=int, default=0,')
src = src.replace('print(f"[code] {n_prot:,} weights ({100*n_prot/tot:.2f}%) unflippable", flush=True)',
                  'print(f"[code] {n_prot:,} weights ({100*n_prot/tot:.2f}%) unflippable, placed by {a.code_score} score", flush=True)')
g = {"__name__": "__main__", "__file__": os.path.join(HERE, "lean_llm_showcase.py")}
exec(compile(src, "lean_llm_showcase.py(codesel)", "exec"), g)
