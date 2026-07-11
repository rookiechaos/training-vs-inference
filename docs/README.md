# Background theory (optional reading)

The root README is oriented toward install + quick start. Longer notes on hardware
non-associativity, LLM symptoms, MLLM visual-anchor drift, and video chaotic
amplification live in:

- [`docs/wan21_m3_practice.md`](wan21_m3_practice.md) — M3 Wan playbook
- [`docs/wan_tim_code_structure.md`](wan_tim_code_structure.md) — code map for video TIM

For LLM-side demos, start with:

```bash
tim demo-reduction
python experiments/01_fp_non_associativity.py
python experiments/02_ppo_tim_simulation.py
```
