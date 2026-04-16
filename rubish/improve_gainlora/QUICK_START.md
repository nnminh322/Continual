# Quick Reference: Run SpecRoute Llama in 10 Commands

## From clean H100 server (first time setup)

```bash
# 1. Go to project
cd /path/to/improve_gainlora

# 2. Create isolated environment
python3.10 -m venv venv_llama_specroute
source venv_llama_specroute/bin/activate

# 3. Install dependencies (one-time, ~5 minutes)
pip install --upgrade pip setuptools wheel
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install deepspeed==0.13.1 transformers==4.36.0 sentencepiece==0.1.99 datasets==2.14.7 nltk==3.8.1 rouge-score==0.1.2 tqdm==4.66.1
pip install --upgrade accelerate
# 4. Download model (first time, ~20 minutes)
python -c "from transformers import LlamaForCausalLM, AutoTokenizer; m = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf'); t = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')"
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True)"

# 5. Quick test (one task, ~3 minutes)
deepspeed --include localhost:0 --master_port 49500 src/run_llama.py \
   --do_train --do_predict --predict_with_generate \
   --model_name_or_path meta-llama/Llama-2-7b-hf \
   --data_dir CL_Benchmark \
   --task_order task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification \
   --task_config_dir configs/gen_script_superni_order1_llama_configs/task1572_samsum_summary \
   --output_dir logs_and_outputs/test_single_task/outputs/1-task1572_samsum_summary \
   --training_epochs 50 --per_device_train_batch_size 2 --per_device_eval_batch_size 4 \
   --lora_r 4 --lora_alpha 32 --threshold 0.995 --model_name specroute --num_train_epochs 50
```

## Subsequent runs (after environment setup)

```bash
# 6. Activate environment
source venv_llama_specroute/bin/activate

# 7. Run full Order 1 (6-10 hours)
nohup bash gen_script_superni_order1_llama_specroute.sh 0 meta-llama/Llama-2-7b-hf > run_order1.log 2>&1 &
tail -f run_order1.log  # Monitor

# 8. Run full Order 2 (6-10 hours)
nohup bash gen_script_superni_order2_llama_specroute.sh 0 meta-llama/Llama-2-7b-hf > run_order2.log 2>&1 &
tail -f run_order2.log  # Monitor

# 9. Calculate metrics
python score.py gen_script_superni_order1_llama_specroute gen_script_superni_order1_llama_specroute
python score.py gen_script_superni_order2_llama_specroute gen_script_superni_order2_llama_specroute

# 10. Compare with baseline
python score.py gen_script_superni_order1_llama_gainlora_inflora gen_script_superni_order1_llama_gainlora_inflora
echo "^^^ This is the GainLoRA baseline to compare with SpecRoute results above ^^^"
```

## Expected Output Example (step 9)

```
[INFO] base_dir: logs_and_outputs
[INFO] run_name: gen_script_superni_order1_llama_specroute
[INFO] task_order.txt: 15 tasks

[INFO] Building cross-task score matrix...

=== Continual Learning Metrics (gen_script_superni_order1_llama_specroute) ===
Cl (Current Learning):    0.451  ← Average on all tasks at end
Fgt (Forgetting):         0.124  ← Average catastrophic forgetting
Fwt (Forward Transfer):   0.424  ← How earlier tasks help future tasks
Bwt (Backward Transfer):  0.087  ← How current learning damages past tasks

=== Cross-Task Score Matrix ===
                   task1572  task363  task1290  ... task875
After task1 :      0.450     0.000    0.000        0.000
After task2 :      0.438     0.462    0.000        0.000
After task3 :      0.435     0.456    0.468        0.000
...
After task15:      0.412     0.440    0.451        0.456
```

## Comparison Example (step 10)

```
GainLoRA InfLoRA (Reference):
Cl: 0.451, Fgt: 0.124, Fwt: 0.424, Bwt: 0.087

SpecRoute (New):
Cl: 0.450, Fgt: 0.125, Fwt: 0.422, Bwt: 0.089

→ Performance highly similar (good! SpecRoute provides parameter-free routing
  without sacrificing accuracy, while being more interpretable via SVD)
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "CUDA out of memory" | Reduce batch size: `--per_device_train_batch_size 1` |
| "score.py not found" | Run from `improve_gainlora/` directory |
| "task_order.txt not found" | Tasks didn't complete; check `tail -100 run_order1.log` |
| NaN loss | Switch to fp32 if bf16 not supported by hardware |
| "Llama-3 not supported" | Use Llama-2-7B or Llama-2-13B for now |

## Environment deactivation

```bash
# When done
deactivate
```

---

**Total time breakdown:**
- Setup (steps 1-4): 40 minutes (one-time)
- Test (step 5): 3 minutes
- Full Order 1 (step 7): 6-10 hours
- Full Order 2 (step 8): 6-10 hours
- Results (steps 9-10): 2 minutes
- **Total: ~13-21 hours of compute time (mostly automated)**

See [SETUP_AND_USAGE_LLAMA_SPECROUTE.md](SETUP_AND_USAGE_LLAMA_SPECROUTE.md) for detailed explanations.
