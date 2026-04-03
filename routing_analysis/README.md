# FGCL — Fisher Geometric Continual LoRA

## CLI Interface

```bash
# ALL: all phases × all models × all benchmarks
python exp_fgcl.py

# Just one phase (all models, all benchmarks)
python exp_fgcl.py --phase T4

# One model, all phases, all benchmarks
python exp_fgcl.py --model test_embeddings/TestBackbone/

# One benchmark, all phases, all models
python exp_fgcl.py --benchmark SuperNI

# Specific: all phases, one model + one benchmark
python exp_fgcl.py --model test_embeddings/TestBackbone/ --benchmark SuperNI
```

Embedding path convention: `{model_path}/{benchmark}/{task}/train.npz`

## Phases

| Phase | Description | Methods | Tasks | Epochs |
|-------|-------------|---------|-------|--------|
| T1 | FSR vs GPM isolation | standard_lora, inflora, fgcl_fsr | 6 | 20 |
| T2 | KF-FNG convergence | standard_lora, fgcl_kfng | 1 | 30 |
| T3 | TAA vs SGR ablation | standard_lora, fgcl_fsr, fgcl_taa, fgcl_sgr | 8 | 20 |
| T4 | Full comparison | all 6 methods | 15 | 20 |

## Methods

| Method | CL Mechanism |
|--------|-------------|
| standard_lora | Plain LoRA (baseline) |
| gainlora | GainLoRA root port (routing + GPM + KL) |
| inflora | InfLoRA (GPM only) |
| fgcl_fsr | LoRA + Fisher Subspace Regularization |
| fgcl_kfng | LoRA + FSR + Kronecker-Factored Fisher NG |
| fgcl_taa | LoRA + FSR + Task Arithmetic Alignment |

## Hypotheses

| H | Claim | Test Phase |
|---|-------|-----------|
| H1 | Fisher subspace protects better than gradient subspace | T1 |
| H2 | KF-FNG converges faster than AdamW | T2 |
| H3 | TAA reduces forgetting more than SGR | T3 |
| H5 | FGCL methods ≥ baselines on AP + BWT | T4 |

## Output

Results saved to `results_fgcl/{model_name}/{benchmark}/phase_{T1,T2,T3,T4}/all_results.json`

Metrics: AP (Average Performance), FT (Forward Transfer), BWT (Backward Transfer)
