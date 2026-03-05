# Benchmark & Metrics Reference — GainLoRA Paper

> Tài liệu này giải thích dataset, metrics, và cách đọc kết quả trong paper GainLoRA.  
> Target audience: người đã biết continual learning nhưng cần nắm nhanh notation.

---

## 1. Datasets

### SuperNI Benchmark (Orders 1 & 2)
**Source:** Super-Natural Instructions (SuperNI) — 1600+ NLP tasks từ nhiều domain.

15 tasks được chọn đại diện cho các loại NLP khác nhau:

| # | Task Name | Loại | Metric |
|---|-----------|------|--------|
| 1 | task1572_samsum_summary | Summarization | RougeL |
| 2 | task363_sst2_polarity_classification | Sentiment | RougeL |
| 3 | task1290_xsum_summarization | Summarization | RougeL |
| 4 | task181_outcome_extraction | Information Extraction | RougeL |
| 5 | task002_quoref_answer_generation | QA | RougeL |
| 6 | task1510_evalution_relation_extraction | Relation Extraction | RougeL |
| 7 | task639_multi_woz_user_utterance_generation | Dialogue Generation | RougeL |
| 8 | task1729_personachat_generate_next | Dialogue Generation | RougeL |
| 9 | task073_commonsenseqa_answer_generation | Commonsense QA | RougeL |
| 10 | task1590_diplomacy_text_generation | Text Generation | RougeL |
| 11 | task748_glucose_reverse_cause_event_detection | Causal Reasoning | RougeL |
| 12 | task511_reddit_tifu_long_text_summarization | Summarization | RougeL |
| 13 | task591_sciq_answer_generation | Science QA | RougeL |
| 14 | task1687_sentiment140_classification | Sentiment | RougeL |
| 15 | task875_emotion_classification | Emotion | RougeL |

**Đặc điểm:** Tasks đa dạng về loại (generation, classification, extraction) → khó giữ nhiều kỹ năng cùng lúc. Metric dùng RougeL (F1 token overlap), scale 0–100.

**Task orders:**
- **Order 1**: Sequential từ summarization → classification (tasks sắp xếp theo domain heterogeneity)
- **Order 2**: Random shuffle để test independence với thứ tự

### Long Benchmark (Orders 3 & 4)
**Source:** Các NLP benchmarks phổ biến (GLUE-style + text classification).

15 tasks:

| # | Task | Loại | Metric |
|---|------|------|--------|
| 1 | yelp | Sentiment (fine-grained) | Exact Match |
| 2 | amazon | Product sentiment | Exact Match |
| 3 | mnli | NLI | Exact Match |
| 4 | cb | NLI (FewShot) | Exact Match |
| 5 | copa | Causal reasoning | Exact Match |
| 6 | qqp | Paraphrase detection | Exact Match |
| 7 | rte | Textual entailment | Exact Match |
| 8 | imdb | Sentiment binary | Exact Match |
| 9 | sst2 | Sentiment binary | Exact Match |
| 10 | dbpedia | Topic classification | Exact Match |
| 11 | agnews | News topic | Exact Match |
| 12 | yahoo | Topic QA | Exact Match |
| 13 | multirc | Multi-sentence RC | Exact Match |
| 14 | boolq | Boolean QA | Exact Match |
| 15 | wic | Word sense disambiguation | Exact Match |

**Đặc điểm:** Nhiều tasks classification → competition giữa các class distributions. Metric dùng Exact Match (%), scale 0–100. Kết quả baseline cao hơn SuperNI (vì tasks ít diverse hơn → ít catastrophic forgetting).

**Task orders:**
- **Order 3**: yelp → wic (ordered từ classification → understanding)
- **Order 4**: mnli → yahoo (scrambled)

---

## 2. Metrics: AP và FT

### AP (Average Performance) ↑

$$AP = \frac{1}{T} \sum_{j=1}^{T} R_{T,j}$$

- $R_{T,j}$ = score trên task $j$ **sau khi đã xong task $T$ (task cuối)**
- Đây là con số **quan trọng nhất** — phản ánh khả năng "nhớ tất cả" sau CL
- **Cao hơn = tốt hơn**
- Baseline chạy 100 epochs (A100), OT-SIGN chạy 50 epochs (T4) → AP ta có thể thấp hơn baseline do epochs ít hơn, không phải do method kém

### FT (Forgetting) ↓

$$FT = \frac{1}{T-1} \sum_{j=1}^{T-1} \left( R_{j,j} - R_{T,j} \right)$$

- $R_{j,j}$ = score trên task $j$ **ngay khi vừa train xong task $j$** (peak performance)
- $R_{T,j}$ = score trên task $j$ **sau khi train hết tất cả** (final performance)
- FT = **trung bình lượng score bị giảm** sau khi học thêm tasks mới
- **Thấp hơn = tốt hơn** (ít forgetting)
- FT = 0 nghĩa là model nhớ hoàn toàn tất cả tasks sau CL

### Result Matrix R

```
         Task1  Task2  Task3  ...  Task15
Train1  [ R11    -      -           -   ]   ← chỉ evaluate task vừa học
Train2  [ R21   R22     -           -   ]
Train3  [ R31   R32    R33          -   ]
...
Train15 [ R151  R152   R153  ...  R1515 ]
         ↑                         ↑
       Final                    Final
       perf                     perf
       task1                    task15
       
Diagonal: R_jj = peak performance (train từng task)
Last row: R_15j = final performance sau CL
AP = mean(last row)
FT = mean(diagonal - last row), j=1..14
```

---

## 3. Phân tích baseline — GainLoRA đứng đâu?

### Table 1 (SuperNI, T5-Large)

**Tốt nhất trước GainLoRA:**
- InfLoRA: AP=39.78, FT=7.64 (Order 1)
- GainLoRA (InfLoRA): AP=**46.21**, FT=**2.40** — cải thiện ~6.4 AP, giảm FT 5x

**Điểm mạnh GainLoRA:**
- FT cực thấp (2.4 vs 7.64 của InfLoRA) → ít forgetting hơn nhiều
- AP cao hơn tất cả methods khác kể cả LFPT5 (39.03)

**Nhận xét về Final Stage:**
- Final stage (task 15) là quan trọng nhất trong deployment
- GainLoRA FT thấp → model vẫn perform tốt trên task 1-14 khi đang làm task 15
- Đây là điểm yếu chính của các method cũ: O-LoRA FT=19.15 nghĩa là mỗi task bị quên trung bình ~19 điểm

### Table 2 (Long, T5-Large)

**Context:**
- Long benchmark dễ hơn → AP cao hơn (70-80% range vs 40-46% range)
- GainLoRA vẫn outperform InfLoRA: AP=78.01 vs 75.15 (Order 3)
- FT cực thấp: 0.77 (gần như không quên!)

---

## 4. OT-SIGN — Cải thiện gì so với GainLoRA baseline?

| Component | GainLoRA | OT-SIGN+GainLoRA |
|-----------|----------|------------------|
| Expert routing | Cosine similarity + sigmoid gating | Sinkhorn OT với vMF signature |
| Continual protection | GPM gradient projection | + Anti-drift (MSE on trans_input) |
| Expert invasion | Không có | + Anti-invasion hinge loss |
| Knowledge repr | Prompt key vector | vMF distribution (mu, kappa) |

### Kỳ vọng:
- **FT nên thấp hơn GainLoRA** (anti-drift + anti-invasion hoạt động đúng)
- **AP tương đương hoặc cao hơn** (OT routing chính xác hơn cosine)
- Nếu AP thấp hơn đáng kể → có thể do epochs ít (50 vs 100), không phải method kém

---

## 5. Cách đọc log và compute_ap_ft.py

### Sau mỗi task hoàn thành, terminal sẽ in:
```
[RunLogger] After task 3 (task1290_xsum_summarization) — predict scores:
  task1572_samsum_summary                         45.23
  task363_sst2_polarity_classification            67.81
  task1290_xsum_summarization                     52.14
```

### Sau task cuối (task 15), terminal sẽ in AP/FT tự động:
```
════════════════════════════════════════════════════════════════════════
  OT-SIGN+GainLoRA Order1
════════════════════════════════════════════════════════════════════════
  #   Task                                             Peak   Final    Drop
────────────────────────────────────────────────────────────────────
  1   task1572_samsum_summary                         48.12   44.30   3.82
  ...
  15  task875_emotion_classification                  71.20   71.20   0.00
────────────────────────────────────────────────────────────────────
  AP  =  47.83   │   FT =  2.15
════════════════════════════════════════════════════════════════════════
```

### Chạy thủ công sau khi xong:
```bash
python src/compute_ap_ft.py \
  --output_base logs_and_outputs/ot_sign_order1_t5large/outputs \
  --task_order "task1572_samsum_summary,..." \
  --method_name "OT-SIGN+GainLoRA Order1" \
  --save
```

Kết quả lưu ở `logs_and_outputs/ot_sign_order1_t5large/ap_ft_result.json`.

---

## 6. Thời gian ước tính trên 2×T4

| Script | Order | Benchmark | Epochs | Tasks | Ước tính |
|--------|-------|-----------|--------|-------|----------|
| run_ot_sign_order1_t5large.sh | 1 | SuperNI | 50 | 15 | ~9-10h |
| run_ot_sign_order2_t5large.sh | 2 | SuperNI | 50 | 15 | ~9-10h |
| run_ot_sign_order3_t5large.sh | 3 | Long | 10 | 15 | ~3-4h |
| run_ot_sign_order4_t5large.sh | 4 | Long | 10 | 15 | ~3-4h |

**Chạy song song:** Order 1+3 trên GPU 0,1 và Order 2+4 trên GPU 0,1 (2 server riêng)  
**Chạy tuần tự:** ~22-28h tổng, khuyến nghị chạy Order 3+4 trước (nhanh hơn, validate pipeline)

---

## 7. FAQ

**Q: rougeL range là gì?**  
A: 0–100 sau multiply (compute_metrics trả về 0-1, code đã nhân 100x). Trên SuperNI, range typical là 20-80.

**Q: GainLoRA chạy 100 epochs trên A100, mình chạy 50 epochs trên T4 — có fair không?**  
A: Không hoàn toàn fair về tuyệt đối. Nhưng mục tiêu là thấy delta: nếu OT-SIGN+GainLoRA(50ep/T4) > GainLoRA(50ep/T4) thì contribution rõ ràng. Để so sánh với paper, cần chạy cùng setups.

**Q: FT âm có nghĩa gì?**  
A: Score cuối cao hơn peak → model "cải thiện" task cũ nhờ học tasks mới (positive transfer). Hiếm nhưng có thể xảy ra với tasks liên quan nhau.

**Q: AP thấp hơn baseline dù FT tốt hơn?**  
A: Có thể do peak performance của từng task thấp → R[j,j] thấp → final row cũng thấp. Nghĩa là OT routing có thể gây interference lúc học task đầu tiên. Kiểm tra peak scores từng task để diagnose.
