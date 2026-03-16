# GainLoRA vs OT-SIGN Results — Direct Comparison

> **Cách đọc:** AP↑ (cao hơn tốt hơn) | FT↓ (thấp hơn tốt hơn)  
> Số trong bảng đều là **rougeL (%)** cho superni / **exact_match (%)** cho long.  
> Lấy kết quả từ `compute_ap_ft.py` sau mỗi lần chạy xong 15 tasks.

---

## Table 1: T5-Large — SuperNI Benchmark (Orders 1 & 2)

| Method | Order 1 AP↑ | Order 1 FT↓ | Order 2 AP↑ | Order 2 FT↓ |
|--------|-------------|-------------|-------------|-------------|
| LFPT5* | 39.03 | 10.87 | 29.70 | 20.72 |
| EWC | 15.32 | 26.78 | 18.19 | 30.28 |
| TaSL | 27.51 | 18.53 | 28.05 | 17.39 |
| KIFLoRA | 28.33 | 16.44 | 30.31 | 16.27 |
| SeqLoRA | 7.30 | 47.60 | 7.03 | 47.97 |
| IncLoRA | 12.33 | 41.93 | 16.65 | 36.56 |
| C-LoRA | 22.69 | 24.25 | 32.81 | 11.60 |
| O-LoRA | 26.37 | 19.15 | 32.83 | 11.99 |
| InfLoRA | 39.78 | 7.64 | 39.57 | 8.93 |
| **GainLoRA (InfLoRA)** | **46.21** | **2.40** | **46.44** | **2.61** |
| **OT-SIGN+GainLoRA (ours)** | | | | |

## Table 2: T5-Large — Long Benchmark (Orders 3 & 4)

| Method | Order 3 AP↑ | Order 3 FT↓ | Order 4 AP↑ | Order 4 FT↓ |
|--------|-------------|-------------|-------------|-------------|
| EPI* | — | — | 75.19 | 0.77 |
| MIGU+FT | — | — | 71.30 | 11.39 |
| EWC | 43.24 | 23.66 | 46.25 | 32.90 |
| TaSL | 71.37 | 6.20 | 73.11 | 6.52 |
| KIFLoRA | 72.19 | 3.10 | 73.72 | 4.75 |
| SeqLoRA | 49.46 | 27.60 | 33.81 | 45.53 |
| IncLoRA | 61.19 | 13.63 | 62.46 | 15.92 |
| C-LoRA | 66.83 | 8.64 | 61.86 | 14.18 |
| O-LoRA | 70.98 | 3.69 | 71.21 | 4.03 |
| InfLoRA | 75.15 | 4.19 | 75.79 | 3.47 |
| **GainLoRA (InfLoRA)** | **78.01** | **0.77** | **77.54** | **1.20** |
| **OT-SIGN+GainLoRA (ours)** | | | | |

---

---

## Table 3: Llama — SuperNI Benchmark (from GainLoRA paper)

### Llama-2-7B

| Method | Order 1 AP↑ | Order 1 FT↓ | Order 2 AP↑ | Order 2 FT↓ |
|--------|-------------|-------------|-------------|-------------|
| O-LoRA | 39.37 | 15.84 | 37.55 | 20.23 |
| GainLoRA (O-LoRA) | 51.10 | 4.96 | 51.14 | 5.57 |
| InfLoRA | 42.93 | 11.23 | 39.94 | 15.00 |
| **GainLoRA (InfLoRA)** | **51.27** | **2.84** | **50.17** | **4.71** |
| **SpecRoute (ours)** | | | | |

### Llama-2-13B

| Method | Order 1 AP↑ | Order 1 FT↓ | Order 2 AP↑ | Order 2 FT↓ |
|--------|-------------|-------------|-------------|-------------|
| O-LoRA | 43.92 | 14.15 | 40.05 | 19.53 |
| GainLoRA (O-LoRA) | 52.47 | 4.78 | 51.68 | 5.86 |
| InfLoRA | 43.64 | 14.85 | 45.74 | 10.61 |
| **GainLoRA (InfLoRA)** | **53.64** | **2.87** | **52.46** | **4.90** |
| **SpecRoute (ours)** | | | | |

### Llama-3-8B

| Method | Order 1 AP↑ | Order 1 FT↓ | Order 2 AP↑ | Order 2 FT↓ |
|--------|-------------|-------------|-------------|-------------|
| O-LoRA | 42.49 | 8.85 | 38.67 | 19.28 |
| GainLoRA (O-LoRA) | 53.39 | 3.56 | 51.69 | 6.20 |
| InfLoRA | 43.27 | 6.02 | 48.77 | 5.88 |
| **GainLoRA (InfLoRA)** | **52.18** | **1.40** | **52.48** | **4.21** |
| **SpecRoute (ours)** | | | | |

---

## Table 4: Ablation Study — GainLoRA with T5-Large & Llama-2-7B (from paper)

| Method | T5-Large O1 AP↑ | T5-Large O1 FT↓ | T5-Large O2 AP↑ | T5-Large O2 FT↓ | Llama-2-7B O1 AP↑ | Llama-2-7B O1 FT↓ | Llama-2-7B O2 AP↑ | Llama-2-7B O2 FT↓ |
|--------|---|---|---|---|---|---|---|---|
| GainLoRA (O-LoRA) | 47.84 | 2.26 | 46.84 | 2.91 | 51.10 | 4.96 | 51.14 | 5.57 |
| No Init Constraints | 35.30 | 17.19 | 39.82 | 12.90 | 44.02 | 11.71 | 42.89 | 14.77 |
| No Update Constraints | 23.01 | 30.32 | 24.96 | 28.14 | 33.74 | 23.06 | 34.71 | 22.36 |
| No Constraints | 26.32 | 26.00 | 30.63 | 22.37 | 34.48 | 23.46 | 36.87 | 21.24 |
| GainLoRA (InfLoRA) | 46.21 | 2.40 | 46.44 | 2.61 | 51.27 | 2.84 | 50.17 | 4.71 |
| No Init Constraints | 45.38 | 3.40 | 43.05 | 5.15 | 50.48 | 3.48 | 48.17 | 6.45 |
| No Update Constraints | 37.69 | 10.94 | 38.85 | 9.31 | 48.52 | 5.68 | 47.85 | 7.00 |
| No Constraints | 36.75 | 12.18 | 41.00 | 6.66 | 49.10 | 6.07 | 45.77 | 8.70 |

> **Note**: "Init Constraints" = LoRA_A null-space projection (GPM), "Update Constraints" = GainLoRA gating + prompt_key routing

---

## Per-Task Breakdown — Order 1 (fill after running)

Chạy lệnh sau để lấy số điền vào:
```bash
python src/compute_ap_ft.py \
  --output_base logs_and_outputs/ot_sign_order1_t5large/outputs \
  --task_order "task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification" \
  --method_name "OT-SIGN+GainLoRA Order1"
```

| # | Task | GainLoRA Peak | GainLoRA Final | OT-SIGN Peak | OT-SIGN Final |
|---|------|--------------|---------------|-------------|--------------|
| 1 | task1572_samsum_summary | | | | |
| 2 | task363_sst2_polarity_classification | | | | |
| 3 | task1290_xsum_summarization | | | | |
| 4 | task181_outcome_extraction | | | | |
| 5 | task002_quoref_answer_generation | | | | |
| 6 | task1510_evalution_relation_extraction | | | | |
| 7 | task639_multi_woz_user_utterance_generation | | | | |
| 8 | task1729_personachat_generate_next | | | | |
| 9 | task073_commonsenseqa_answer_generation | | | | |
| 10 | task1590_diplomacy_text_generation | | | | |
| 11 | task748_glucose_reverse_cause_event_detection | | | | |
| 12 | task511_reddit_tifu_long_text_summarization | | | | |
| 13 | task591_sciq_answer_generation | | | | |
| 14 | task1687_sentiment140_classification | | | | |
| 15 | task875_emotion_classification | | | | |
| | **AP / FT** | **46.21 / 2.40** | | | |

## Per-Task Breakdown — Order 2

```bash
python src/compute_ap_ft.py \
  --output_base logs_and_outputs/ot_sign_order2_t5large/outputs \
  --task_order "task748_glucose_reverse_cause_event_detection,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task639_multi_woz_user_utterance_generation,task1572_samsum_summary,task1687_sentiment140_classification,task591_sciq_answer_generation,task363_sst2_polarity_classification,task1510_evalution_relation_extraction,task1729_personachat_generate_next,task181_outcome_extraction,task511_reddit_tifu_long_text_summarization,task002_quoref_answer_generation,task1290_xsum_summarization,task875_emotion_classification" \
  --method_name "OT-SIGN+GainLoRA Order2"
```

| # | Task | GainLoRA Peak | GainLoRA Final | OT-SIGN Peak | OT-SIGN Final |
|---|------|--------------|---------------|-------------|--------------|
| 1 | task748_glucose_reverse_cause_event_detection | | | | |
| 2 | task073_commonsenseqa_answer_generation | | | | |
| 3 | task1590_diplomacy_text_generation | | | | |
| 4 | task639_multi_woz_user_utterance_generation | | | | |
| 5 | task1572_samsum_summary | | | | |
| 6 | task1687_sentiment140_classification | | | | |
| 7 | task591_sciq_answer_generation | | | | |
| 8 | task363_sst2_polarity_classification | | | | |
| 9 | task1510_evalution_relation_extraction | | | | |
| 10 | task1729_personachat_generate_next | | | | |
| 11 | task181_outcome_extraction | | | | |
| 12 | task511_reddit_tifu_long_text_summarization | | | | |
| 13 | task002_quoref_answer_generation | | | | |
| 14 | task1290_xsum_summarization | | | | |
| 15 | task875_emotion_classification | | | | |
| | **AP / FT** | **46.44 / 2.61** | | | |

## Per-Task Breakdown — Order 3 (Long)

```bash
python src/compute_ap_ft.py \
  --output_base logs_and_outputs/ot_sign_order3_t5large/outputs \
  --task_order "yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic" \
  --method_name "OT-SIGN+GainLoRA Order3"
```

| # | Task | GainLoRA Peak | GainLoRA Final | OT-SIGN Peak | OT-SIGN Final |
|---|------|--------------|---------------|-------------|--------------|
| 1 | yelp | | | | |
| 2 | amazon | | | | |
| 3 | mnli | | | | |
| 4 | cb | | | | |
| 5 | copa | | | | |
| 6 | qqp | | | | |
| 7 | rte | | | | |
| 8 | imdb | | | | |
| 9 | sst2 | | | | |
| 10 | dbpedia | | | | |
| 11 | agnews | | | | |
| 12 | yahoo | | | | |
| 13 | multirc | | | | |
| 14 | boolq | | | | |
| 15 | wic | | | | |
| | **AP / FT** | **78.01 / 0.77** | | | |

## Per-Task Breakdown — Order 4 (Long)

```bash
python src/compute_ap_ft.py \
  --output_base logs_and_outputs/ot_sign_order4_t5large/outputs \
  --task_order "mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo" \
  --method_name "OT-SIGN+GainLoRA Order4"
```

| # | Task | GainLoRA Peak | GainLoRA Final | OT-SIGN Peak | OT-SIGN Final |
|---|------|--------------|---------------|-------------|--------------|
| 1 | mnli | | | | |
| 2 | cb | | | | |
| 3 | wic | | | | |
| 4 | copa | | | | |
| 5 | qqp | | | | |
| 6 | boolq | | | | |
| 7 | rte | | | | |
| 8 | imdb | | | | |
| 9 | yelp | | | | |
| 10 | amazon | | | | |
| 11 | sst2 | | | | |
| 12 | dbpedia | | | | |
| 13 | agnews | | | | |
| 14 | multirc | | | | |
| 15 | yahoo | | | | |


```bash
# Chạy 4 lệnh này để lấy đủ số cho cả 2 bảng:
python src/compute_ap_ft.py --output_base logs_and_outputs/ot_sign_order1_t5large/outputs --task_order "task1572_samsum_summary,task363_sst2_polarity_classification,task1290_xsum_summarization,task181_outcome_extraction,task002_quoref_answer_generation,task1510_evalution_relation_extraction,task639_multi_woz_user_utterance_generation,task1729_personachat_generate_next,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task748_glucose_reverse_cause_event_detection,task511_reddit_tifu_long_text_summarization,task591_sciq_answer_generation,task1687_sentiment140_classification,task875_emotion_classification" --save

python src/compute_ap_ft.py --output_base logs_and_outputs/ot_sign_order2_t5large/outputs --task_order "task748_glucose_reverse_cause_event_detection,task073_commonsenseqa_answer_generation,task1590_diplomacy_text_generation,task639_multi_woz_user_utterance_generation,task1572_samsum_summary,task1687_sentiment140_classification,task591_sciq_answer_generation,task363_sst2_polarity_classification,task1510_evalution_relation_extraction,task1729_personachat_generate_next,task181_outcome_extraction,task511_reddit_tifu_long_text_summarization,task002_quoref_answer_generation,task1290_xsum_summarization,task875_emotion_classification" --save

python src/compute_ap_ft.py --output_base logs_and_outputs/ot_sign_order3_t5large/outputs --task_order "yelp,amazon,mnli,cb,copa,qqp,rte,imdb,sst2,dbpedia,agnews,yahoo,multirc,boolq,wic" --save

python src/compute_ap_ft.py --output_base logs_and_outputs/ot_sign_order4_t5large/outputs --task_order "mnli,cb,wic,copa,qqp,boolq,rte,imdb,yelp,amazon,sst2,dbpedia,agnews,multirc,yahoo" --save
```

---

## Table 3: T5-Small — Long Benchmark (Order 3)

| Method | Order 3 AP↑ | Order 3 FT↓ |
|--------|-------------|-------------|
| **GainLoRA (Root)** | **59.70** | N/A* |
| **SpecRoute (Improve)** | 39.74† | N/A* |

> *\*FT = N/A: cả 2 log chạy thiếu `--do_predict`. Lần tiếp theo dùng script `T5_small/` đã sửa sẽ có đủ FT.*  
> *†Điểm Improve tính từ `predict_eval_predictions.jsonl` của từng task (hàng chéo score matrix). imdb/sst2/wic về 0 do Catastrophic Forgetting.*

### ⚠️ Root GainLoRA tốt hơn SpecRoute trên T5-Small (−19.96 AP)

SpecRoute bị Catastrophic Forgetting nghiêm trọng ở các task phân loại sentiment (imdb=0.21, sst2=0.00, yahoo=8.12, wic=0.00). Nguyên nhân có thể do SVD rank không đủ lớn ở T5-Small, làm routing mechanism không phân tách được subspace của các task.

## Per-Task Breakdown — Order 3 (T5-Small)

| # | Task | GainLoRA (Root) | SpecRoute (Improve) | Δ (Improve−Root) |
|---|------|-----------------|--------------------|-----------------|
| 1 | yelp | 56.01 | 54.36 | −1.65 |
| 2 | amazon | 52.05 | 50.01 | −2.04 |
| 3 | mnli | 34.07 | 35.50 | +1.43 |
| 4 | cb | 3.57 | 0.00 | −3.57 |
| 5 | copa | 42.00 | 44.00 | +2.00 |
| 6 | qqp | 76.96 | 76.72 | −0.24 |
| 7 | rte | 45.85 | 50.90 | +5.05 |
| 8 | imdb | 89.51 | 0.21 | **−89.30 ⚠️** |
| 9 | sst2 | 85.21 | 0.00 | **−85.21 ⚠️** |
| 10 | dbpedia | 98.16 | 92.22 | −5.94 |
| 11 | agnews | 88.37 | 68.76 | −19.61 |
| 12 | yahoo | 57.28 | 8.12 | **−49.16 ⚠️** |
| 13 | multirc | 50.52 | 54.23 | +3.71 |
| 14 | boolq | 60.43 | 61.13 | +0.70 |
| 15 | wic | 55.49 | 0.00 | **−55.49 ⚠️** |
| | **AP / FT** | **59.70 / N/A** | **39.74 / N/A** | **−19.96** |

