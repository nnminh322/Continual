# Routing Analysis — Insights

---

# Phase B/C Routing Accuracy Analysis

**Backbone**: flan-t5-large (d_model = 1024)  
**Benchmark**: Long_Sequence (15 tasks)  
**Hai runs**: raw embeddings vs. ZCA-whitened embeddings  
**Files**: `routing_flan-t5-large_Long_Sequence.json` / `_whitened.json`

---

## R1 — Kết quả tổng quan: routing accuracy theo method

| Method | Raw acc | Whitened acc | Δ |
|--------|---------|--------------|---|
| **L2 (centroid)** | 96.74% | **99.99%** | +3.25pp |
| **Cosine / NormL2** | 97.14% | 99.96% | +2.82pp |
| **Mahalanobis** | **99.98%** | **99.99%** | ≈0 |
| SpectralAffinity | 50.85% | 96.52% | +45.67pp |
| SubspaceResidual | 50.85% | 96.52% | +45.67pp |
| WeightedSpectral | 31.22% | 97.06% | +65.84pp |
| PSR_full | 81.15% | **15.25%** | **−65.9pp** |
| PSR_no_mean | 0.63% | 0.82% | ≈0 |
| PSR_no_subspace | 93.50% | 72.93% | −20.57pp |
| PSR_no_penalty | 66.98% | 2.29% | −64.69pp |
| **LDA** | 99.99% | 99.99% | ≈0 |
| **LinearSVM** | **99.99%** | **99.99%** | ≈0 |
| **RidgeClassifier** | 99.99% | 99.97% | ≈0 |
| QDA | 99.08% | **40.45%** | **−58.63pp** |

---

## R2 — CRITICAL: PSR_full sụp đổ sau khi whitening (15.25%)

Đây là kết quả quan trọng và **đáng ngạc nhiên nhất**.

**Dữ liệu chi tiết PSR ablation:**

| Variant | Raw | Whitened | Diễn giải |
|---------|-----|----------|-----------|
| PSR_full | 81.15% | **15.25%** | Cả mean + subspace |
| PSR_no_mean | 0.63% | 0.82% | Chỉ subspace, không mean |
| PSR_no_subspace | 93.50% | 72.93% | Chỉ mean, không subspace |
| PSR_no_penalty | 66.98% | 2.29% | Không có penalty term |

**Phân tích nguyên nhân PSR_full thất bại sau whitening:**

1. **PSR dựa vào anisotropy của raw space**: Công thức PSR score sử dụng principal subspace của task. Trong raw space, mỗi task có các "hướng quan trọng" (dominant eigenvectors) phản ánh đặc trưng domain. ZCA whitening **xóa bỏ hoàn toàn sự khác biệt về độ lớn** của các eigenvectors — các PC của mỗi task trong whitened space đều có variance = 1, không còn "dominant direction" nữa.

2. **Trong whitened space, principal subspace của task không còn discriminative**: Từ Phase A I3, sau whitening tất cả 105 cặp subspace đều gần trực giao. Điều này tốt cho separation theo hướng mean, nhưng **xấu cho PSR**: chọn task theo ||V_t^T x||² trong môi trường mà các V_t đều "random" sẽ cho kết quả gần random.

3. **Mean (centroid) là thứ thực sự discriminative sau whitening**: PSR_no_subspace (chỉ mean) = 72.93% sau whitening, còn L2 centroid = 99.99%. Sự chênh lệch là do PSR_no_subspace vẫn dùng formula PSR (không phải thuần túy nearest-centroid L2).

**Kết luận**: **PSR không tương thích với whitening**. Đây là design conflict: PSR được thiết kế cho raw transformer embedding với strong anisotropy. Whitening triệt tiêu nguồn tín hiệu mà PSR khai thác.

---

## R3 — Whitening biến L2 centroid thành gần như hoàn hảo (99.99%)

**Trong raw space**, L2 = 96.74% — đủ tốt nhưng không xuất sắc. Task yếu nhất:
- rte: 85.92%, boolq: 88.29%, amazon: 94.49%

**Sau whitening**, L2 = 99.99%. Hầu hết task đạt 100%, chỉ có:
- boolq: 99.94%, agnews: 99.99%

**Giải thích**: Whitening chuẩn hóa scale của tất cả dimensions → L2 distance trở thành Mahalanobis distance (đo khoảng cách theo covariance-normalized space). **L2 trên whitened embedding = Mahalanobis trên raw embedding** về mặt toán học.

**Implication cho CL router**: Nearest-centroid với ZCA whitening là router tối ưu cho cả **chất lượng (99.99%)** lẫn **tính incremental** (O(d) update per sample: chỉ cần cập nhật running mean và covariance).

---

## R4 — Mahalanobis = Upper bound thực sự (~99.99% cả hai space)

Mahalanobis đạt 99.98% (raw) và 99.99% (whitened) — **luôn gần hoàn hảo bất kể có whitening hay không**. Điều này xác nhận:

1. Task embeddings **linearly separable hoàn toàn** trong không gian 1024-dim
2. Covariance structure của từng task đủ khác nhau để phân biệt tốt
3. Mahalanobis là "oracle baseline" — biết full covariance matrix của mỗi task

Điểm yếu của Mahalanobis trong CL: cần lưu N×N covariance (hoặc dùng Woodbury incremental). L2 + whitening đơn giản hơn nhiều với kết quả tương đương.

---

## R5 — SpectralAffinity / SubspaceResidual cải thiện mạnh sau whitening

| Method | Raw | Whitened |
|--------|-----|----------|
| SpectralAffinity | 50.85% | **96.52%** |
| SubspaceResidual | 50.85% | **96.52%** |
| WeightedSpectral | 31.22% | **97.06%** |

Trong **raw space**, các phương pháp dựa trên spectral/subspace bị hỏng hoàn toàn (~50% = random với 15 tasks = 1/15 = 6.67%... thực ra 50% cho thấy chúng đang bắt được gì đó nhưng bị nhầm lẫn).

Sau **whitening**, các phương pháp này nhảy lên 96–97%. Điều này nhất quán với I3: sau whitening, subspace structure trở nên clean và discriminative cho spectral methods.

**Nhưng vẫn kém hơn L2 centroid (99.99%)** — subspace-based routing sau whitening không vượt được centroid đơn giản.

---

## R6 — Phase C: Task embeddings linearly separable hoàn toàn

| Classifier | Raw | Whitened |
|------------|-----|----------|
| LDA | **99.99%** | **99.99%** |
| LinearSVM | **99.99%** | **99.99%** |
| RidgeClassifier | **99.99%** | **99.97%** |
| QDA | 99.08% | **40.45%** |

**Kết luận Phase C**:
- **LDA / LinearSVM / Ridge đạt ~100% trong cả hai không gian** → Task routing là **bài toán linearly separable** với flan-t5-large embeddings. Đây là tin rất tốt cho PSR vì PSR là linear router.
- **QDA sụp đổ sau whitening (40.45%)**: QDA dùng per-class full covariance. Sau whitening, các class covariance trở nên giống nhau (gần identity) → QDA không còn thêm thông tin so với LDA → nhưng vẫn fit nhiều tham số hơn → overfitting / numerical issues. Nhất quán với I6: whitening phá vỡ per-class Gaussian assumption.

---

## R7 — Per-task analysis: task nào khó nhất?

**Trong raw space, task khó nhất:**

| Task | L2 (raw) | Cosine (raw) | PSR_full (raw) |
|------|----------|--------------|----------------|
| rte | 85.92% | 83.03% | 79.06% |
| boolq | 88.29% | 92.99% | 78.75% |
| amazon | 94.49% | 92.61% | 79.34% |
| yelp | **96.00%** | 96.43% | **48.12%** |

**yelp trong PSR_full** = 48.12% — gần như random! Trong raw space, yelp và amazon có subspace rất gần nhau (I3: frobenius overlap = 3.84), PSR bị nhầm lẫn giữa 2 task này. Đây là điểm yếu cụ thể của PSR với task đồng domain.

**Sau whitening, hầu hết task đạt 100% với L2/Cosine/Mahalanobis** → whitening xóa nhầm lẫn inter-domain.

---

## R8 — Top-5 ranking tổng hợp

**Raw space top-5:**
1. LinearSVM: 99.99%
2. LDA: 99.99%
3. RidgeClassifier: 99.99%
4. Mahalanobis: 99.98%
5. QDA: 99.08%

**Whitened space top-5:**
1. LinearSVM: 99.99%
2. RidgeClassifier: 99.97%
3. L2: **99.99%**
4. Mahalanobis: 99.99%
5. LDA: 99.99%

**Quan sát**: Sau whitening, **L2 centroid vào top-5** (không cần sklearn classifier) ngang hàng LinearSVM/Mahalanobis. Đây là lợi thế lớn cho CL vì L2 centroid fully incremental.

---

## R9 — Khuyến nghị kiến trúc router cho CL

| Tiêu chí | Best method | Acc |
|----------|------------|-----|
| **Incremental + Simple** | L2 centroid + ZCA whitening | 99.99% |
| **Incremental + No whitening** | L2 centroid / Cosine centroid | 96.74–97.14% |
| **Batch oracle** | LinearSVM / LDA (whitened or raw) | 99.99% |
| **Tránh** | PSR_full + whitening | 15.25% |
| **Tránh** | QDA + whitening | 40.45% |

**Recommended design cho CL router**: `ZCA whiten → L2 centroid (nearest-centroid)`
- Update O(d) per sample: cập nhật running mean và running covariance (Woodbury)
- Không cần train sklearn classifier
- Đạt accuracy ngang Mahalanobis và LDA

---

## R10 — Kết luận về flag `--skip_sklearn`

**Có cần chạy thêm Phase C không?**

Phase C đã chạy và cho kết quả rõ ràng:
- **LDA/LinearSVM/Ridge ≈ 100% trong cả hai space** → đây là **upper bound thực sự** của bài toán routing
- **QDA + whitened = 40.45%** → kết quả đáng chú ý cho paper
- **Không có gì mới để khám phá** từ Phase C với backbone/benchmark này

**Verdict**: Với Long_Sequence + flan-t5-large, **đã có đủ data Phase C** để kết luận. Dùng `--skip_sklearn` cho các backbone/benchmark khác để tiết kiệm thời gian — chỉ cần Phase B metrics để so sánh. Chạy Phase C một lần nữa nếu reviewer yêu cầu ablation trên SuperNI hoặc LLaMA backbone.

---

## R11 — Summary: 5 insight đáng giá nhất từ Phase B/C

1. **PSR không tương thích với whitening**: PSR_full raw=81% → whitened=15%. Không dùng PSR sau ZCA whitening.

2. **L2 centroid + whitening = near-perfect router** (99.99%), đơn giản, fully incremental, không cần classifier.

3. **Mahalanobis = L2 + whitening**: cả hai đều ~99.99% — xác nhận lý thuyết (Mahalanobis ≡ L2 trong whitened space).

4. **Task routing là bài toán linearly separable 100%** với flan-t5-large: LDA/LinearSVM đều đạt ~100% → PSR không cần subspace trick trong không gian whitened.

5. **Subspace-based phương pháp phải chọn đúng không gian**: SpectralAffinity raw=50% nhưng whitened=96.5%. PSR ngược lại: raw=81%, whitened=15%. **Không có một phương pháp nào "domain-agnostic"** — whitening là quyết định thiết kế quan trọng.

---

# Phase A Geometry Analysis — Insights

**Backbone**: flan-t5-large (d_model = 1024)  
**Benchmark**: Long_Sequence (15 tasks)  
**Subspace dim**: k = 8  
**Two runs**: raw embeddings vs. ZCA-whitened embeddings

---

## I1 — Cực kỳ anisotropic trong không gian raw (transformer isotropy problem)

**Dữ liệu:**

| Task | condition_number (raw) | anisotropy_ratio (raw) | condition_number (whitened) |
|------|------------------------|------------------------|----------------------------|
| yelp | 315.5 | 1.5 × 10⁹ | ~1–6 |
| amazon | 142.4 | 7.2 × 10⁸ | ~1–6 |
| copa | 40.1 | 1.25 × 10¹³ | 15.2 |
| cb | 93.5 | 3.1 × 10¹³ | 20.3 |
| qqp | 201.0 | 1.5 × 10⁹ | ~1–6 |
| rte | 146.5 | 1.1 × 10⁹ | ~1–6 |
| multirc | 109.2 | 3.2 × 10⁹ | ~1–6 |

**Kết luận**: Trong không gian raw, tất cả 15 task đều có condition_number rất cao (40–315), isotropy_score ≈ 0. Điều này xác nhận hiện tượng **representation degeneration** phổ biến của transformer: các embedding tập trung gần một không gian con hẹp, bị chi phối bởi 1–2 chiều chính. ZCA whitening khắc phục triệt để vấn đề này (condition_number giảm xuống 1.9–6.6 cho hầu hết task).

**cb và copa** (n_samples nhỏ: 250 và 400) vẫn có condition_number cao sau whitening (20.3 và 15.2) vì quá ít mẫu để ước lượng covariance đủ ổn định.

---

## I2 — Whitening mở rộng effective dimensionality gấp 10–20 lần

**Dữ liệu:**

| Task | n_train | participation_ratio (raw) | participation_ratio (white) | effective_rank (raw) | effective_rank (white) |
|------|---------|---------------------------|-----------------------------|----------------------|------------------------|
| yelp | 10000 | 9.9 | 281 | 56 | 415 |
| amazon | 10000 | 15.3 | 313 | 62 | 447 |
| dbpedia | 14000 | 12.8 | 668 | 62 | 799 |
| yahoo | 10000 | 20.3 | 626 | 74 | 742 |
| mnli | 10000 | 14.7 | 327 | 64 | 461 |
| rte | 2490 | 18.6 | 260 | 82 | 377 |
| cb | 250 | 33.6 | 85 | 111 | 156 |
| copa | 400 | 50.2 | 113 | 151 | 196 |

**Kết luận**: Trước whitening, embedding chỉ thực sự dùng ~10–50 chiều trong không gian 1024-dim (participation ratio 1–5%). Sau whitening, con số này tăng lên 85–668 (8–65%). Điều này cho thấy **whitening không chỉ chuẩn hóa scale mà còn giải phóng thông tin tiềm ẩn** bị che khuất bởi các chiều dominant.

Task lớn (dbpedia, yahoo) được hưởng lợi nhiều nhất vì có đủ mẫu để ước lượng covariance chính xác. Task nhỏ (cb N=250, copa N=400) vẫn bị hạn chế bởi sample size.

---

## I3 — CRITICAL: Sau whitening, tất cả subspace của 15 task gần như TRỰC GIAO nhau

Đây là insight quan trọng nhất của toàn bộ Phase A.

**Dữ liệu A4 — Subspace distances (k=8 principal subspace):**

| Metric | Raw (tầm giá trị) | Max lý thuyết | Whitened (tầm giá trị) |
|--------|-------------------|---------------|------------------------|
| geodesic | 2.5 – 3.9 | π/2·√8 ≈ **4.44** | **4.2 – 4.4** |
| chordal | 2.0 – 2.7 | √8 ≈ **2.83** | **2.78 – 2.83** |
| projection | 2.9 – 3.9 | k = **8.0** | **4.0 – 4.3** |
| frobenius_overlap | 0.44 – 3.84 | — | **0.001 – 0.1** |

**Giải thích**: Trong không gian raw:
- Các cặp task tương đồng (yelp-amazon: geodesic=2.70, yelp-imdb: 2.84, amazon-imdb: 2.53) có subspace khá gần nhau — **chúng dùng chung không gian con sentimentl**.
- Frobenius overlap yelp-amazon = 3.84 (rất cao), cho thấy hai task này hầu như dùng cùng 8 principal directions.

Sau ZCA whitening:
- **Tất cả 105 cặp** (15 tasks chọn 2) có geodesic ≈ 4.2–4.4, tức gần max 4.44.
- frobenius_overlap giảm xuống còn 0.001–0.1 cho hầu hết cặp.
- **Subspace của mọi task đều gần như trực giao nhau trong không gian whitened.**

**Ý nghĩa cho PSR routing**: PSR (Principal Subspace Routing) hoạt động bằng cách chiếu embedding vào subspace của từng task và tìm task có projection lớn nhất. Nếu subspace không trực giao, cách đo này bị nhiễu. Sau whitening, các task subspace gần như trực giao → **PSR có nền tảng hình học vững chắc**. Đây là biện hộ hình học trực tiếp cho design của phương pháp.

---

## I4 — Whitening phân tách cụm task tương đồng (cluster structure)

**Trong không gian raw**, các task sentiment (yelp, amazon, imdb) có:
- Geodesic subspace distance: 2.53–2.84 (thấp nhất trong toàn tập)
- Frobenius overlap: 3.8 (cực cao)
- Cosine centroid distance: 0.067 (yelp-amazon), 0.137 (yelp-imdb), 0.142 (amazon-imdb)

So sánh với các cặp cross-domain như copa-imdb (geodesic=3.85, cosine=0.59) hay sst2-multirc (cosine=0.68).

**Whitening xóa bỏ cụm này** bằng cách loại shared biases của transformer. Sau whitening, tất cả cặp đều có geodesic ~4.3 — **không còn "cluster" nhân tạo** nữa.

**Implication**: Nếu dùng raw embedding để routing mà không whitening, router sẽ bị nhầm lẫn giữa yelp/amazon/imdb vì chúng dùng chung subspace. Whitening là **bước tiền xử lý bắt buộc** cho subspace-based routing.

---

## I5 — Whitening lộ ra cấu trúc đa phương thức (multi-modal) ẩn

**Dữ liệu A2b — GMM BIC multimodality:**

| Task | best_k (raw) | best_k (whitened) |
|------|-------------|-------------------|
| cb (n=250, 3 classes) | **1** (unimodal!) | 2 |
| copa (n=400, 2 classes) | **1** (unimodal!) | 2 |
| sst2 | 5 | 4 |
| yelp | 5 | 5 |
| dbpedia | 4 | 5 |
| mnli | 5 | 5 |
| boolq | 5 | 5 |
| wic | 4 | 4 |

**Trước whitening**: cb và copa xuất hiện unimodal — cấu trúc multi-class bị che khuất bởi chiều dominant của transformer representation.

**Sau whitening**: Tất cả 15 task đều multi-modal (best_k ≥ 2). Ngay cả cb (3 classes, N=250) cũng hiện best_k=2.

**Kết luận**: **Whitening "lộ ra" cấu trúc per-class** tồn tại trong dữ liệu nhưng bị che bởi anisotropy. Điều này có nghĩa là thông tin discriminative theo label đã có trong embedding nhưng ở không gian bị "ép" vào ít chiều.

---

## I6 — Non-Gaussianity tập trung ở task multi-class sau whitening (hàm ý cho PPCA)

**Dữ liệu A2 — Mean absolute kurtosis:**

| Task | n_classes | kurtosis (raw) | kurtosis (whitened) |
|------|-----------|----------------|---------------------|
| dbpedia | 14 | ~0.5 | **2.85** |
| yahoo | 10 | ~0.4 | **1.64** |
| qqp | 2 | 0.70 | **1.36** |
| boolq | 2 | ~0.4 | **1.09** |
| mnli | 3 | ~0.5 | ~0.6 |
| yelp | 2 | ~0.3 | ~0.3 |
| imdb | 2 | ~0.3 | 0.17 |
| cb | 3 | ~0.4 | ~0.4 |

**Kết luận**: Trong không gian raw, hầu hết task trông như Gaussian (kurtosis < 0.7). Sau whitening, các task **nhiều class** (dbpedia=14, yahoo=10) có kurtosis tăng mạnh (2.85, 1.64), chứng tỏ phân phối thực ra là **mixture of Gaussians** (per-class clustering), không phải một Gaussian.

**Hàm ý**: PSR dùng PPCA (per-task Gaussian) chỉ là xấp xỉ cho single class hoặc binary task. Với multi-class tasks (dbpedia, yahoo), **một mô hình PPCA đơn không đủ**. Giải pháp tiềm năng: routing bằng GMM hoặc per-class subspace.

---

## I7 — Few-shot stability: subspace ổn định từ n=50–100 samples

**Dữ liệu A7 — std_proj_dist tại các mức sample size:**

| Task | std (n=50) | std (n=100) | std (n=200) | std (n=500) |
|------|-----------|------------|------------|------------|
| yelp | 0.078 | 0.143 | 0.147 | 0.231 |
| amazon | 0.106 | 0.082 | 0.123 | 0.075 |
| rte | **0.031** | 0.040 | 0.168 | 0.165 |
| cb | **0.027** | 0.089 | 0.078 | — |
| wic | **0.048** | 0.085 | 0.072 | 0.096 |
| dbpedia | 0.225 | 0.104 | 0.144 | 0.087 |
| boolq | 0.110 | 0.089 | 0.096 | 0.138 |
| sst2 | 0.049 | 0.171 | 0.069 | 0.203 |

*Các giá trị std trên là trong không gian raw (non-whitened). Whitened values tương tự nhưng thấp hơn.*

**Kết luận**: `std_proj_dist` ở n=50 dao động 0.027–0.225 (phần lớn < 0.15), cho thấy:
1. **Subspace ổn định sớm**: đa số task hội tụ principal subspace tốt từ 50–100 samples.
2. `dbpedia` (0.225 ở n=50) là **task khó nhất để ước lượng subspace** vì có 14 classes.
3. `rte`, `cb`, `wic` cực kỳ ổn định vì dữ liệu ít nhưng tập trung.
4. **Continual learning implication**: Router có thể update reliably từ n≥50 samples per task.

---

## I8 — Centroid distances: moderate separability in raw space

**Dữ liệu A3 — Cosine centroid distances (raw):**

- Gần nhất: yelp–amazon (0.068), multirc–boolq (0.082)
- Xa nhất: sst2–multirc (0.675), copa–imdb (0.593)
- Cross-domain (NLI–Sentiment) ~0.4–0.5
- Within-domain (Sentiment) ~0.07–0.14

**L2 centroid distances (raw)**:
- Range: 0.25 (yelp–amazon) đến 1.01 (sst2–multirc)

**Kết luận**: Centroid distances trong raw space phản ánh domain similarity — task thuộc cùng domain (sentiment: yelp, amazon, imdb) có centroid gần nhau. Tuy nhiên, centroid-based routing (NearestCentroid) vẫn có thể nhầm lẫn nếu không whitening. **Cosine centroid sau whitening sẽ phân tách tốt hơn** vì whitening cân bằng scale các chiều.

---

## Summary: Có nên dùng flag `--skip_sklearn` khi chạy bash không?

`--skip_sklearn` là flag của `run_phase_BC.sh` (biến `SKIP_SKLEARN`), kiểm soát việc bỏ qua phần sklearn classifiers (Phase C) trong phase B/C run.

### Kết luận trực tiếp từ data:

**1. Geometric justification cho PSR đã đủ mạnh**: I3 cho thấy sau whitening, mọi task subspace gần như trực giao. PSR có nền tảng hình học vững chắc — không cần bằng chứng empirical bổ sung từ sklearn baseline ngay lập tức.

**2. Phần sklearn vẫn có giá trị cho:**
- Baseline accuracy cho paper (reviewer sẽ yêu cầu so sánh với classifier chuẩn)
- So sánh tốc độ và accuracy: RidgeClassifier / SVM vs PSR trong incremental setting
- Xác nhận PSR không thua trong setting đơn giản

**3. Recommendation**:
- Dùng `--skip_sklearn` (tắt sklearn) cho **quick runs** / sweep trên nhiều backbone-benchmark
- Chạy **không có `--skip_sklearn`** ít nhất 1 lần trên Long_Sequence để lấy baseline đầy đủ cho bảng so sánh
- Priority thấp hơn Phase D (ablation PSR vs RLS) và Phase F (GPM vs PSR)

---

## Khuyến nghị cho các bước tiếp theo

| Priority | Action | Lý do |
|----------|--------|--------|
| HIGH | Chạy Phase F (GPM simulation) | Xác nhận GPM null-space projections không cần thiết khi whitened |
| HIGH | Chạy Phase D (ablation PSR vs RLS) | Compare incremental tracking quality |
| MEDIUM | Chạy `run_phase_BC.sh` không có `--skip_sklearn` | Lấy baseline sklearn accuracy cho paper |
| LOW | Thử nghiệm per-class PPCA cho dbpedia/yahoo | I6 cho thấy single PPCA có thể không đủ với 10–14 classes |
| LOW | Chạy Phase B (cross-backbone) | Kiểm tra xem geometry insights có transfer sang T5-base, LLaMA không |

---

## Kết luận tổng quan

Bốn insight chính từ Phase A:

1. **Whitening là bắt buộc**: Raw transformer embeddings cực kỳ anisotropic (condition_number 40–315). Không whitening, subspace routing bị nhiễu bởi dominant shared directions.

2. **Sau ZCA whitening, tất cả 15 task subspace (k=8) gần như trực giao nhau** — đây là điều kiện lý tưởng để PSR hoạt động. Geometric separability =  empirical motivation for PSR design.

3. **Multi-modal structure tồn tại ở tất cả task** nhưng bị ẩn trong raw space. Whitening "giải phóng" cấu trúc này. Tuy nhiên, task multi-class lớn (dbpedia, yahoo) có non-Gaussianity cao → single PPCA là xấp xỉ.

4. **Subspace ổn định từ 50–100 samples** — router PSR/RLS có thể hoạt động reliable trong CL setting ngay cả khi số mẫu per task nhỏ.

---

---

# Phase D Ablation Analysis — Insights

**Backbone**: flan-t5-large (d_model = 1024)  
**Benchmark**: Long_Sequence (15 tasks)  
**Hai runs**: raw embeddings vs. ZCA-whitened embeddings  
**Files**: `ablation_flan-t5-large_Long_Sequence.json` / `_whitened.json`  
**Sections**: D1 (PSR component ablation), D2 (rank sweep), D3 (domain breakdown), D6 (incremental simulation)

---

## D-I1 — CRITICAL: Centroid_only (raw) > PSR_full (raw) — subspace component làm hại PSR

**Dữ liệu D1 ablation:**

| Variant | Raw acc | Whitened acc | Diễn giải |
|---------|---------|--------------|-----------|
| Centroid_only | **82.92%** | 4.80% | Chỉ dùng mean term |
| PSR_full | 81.15% | 15.25% | Mean + subspace + penalty |
| PSR_light (= PSR_no_penalty) | 66.98% | 2.29% | Mean + subspace, không penalty |
| Subspace_only | 0.54% | 0.004% | Chỉ dùng subspace, không mean |

**Kết luận quan trọng**: Trong **raw space**, `Centroid_only` (82.92%) **vượt** `PSR_full` (81.15%) **1.77pp**. Điều này có nghĩa:

1. **Subspace component trong PSR không mang lại lợi ích net** — nó thực ra làm giảm accuracy so với chỉ dùng centroid thuần túy.
2. **Penalty term cứu PSR**: Không có penalty, PSR_light = 66.98% (thấp hơn cả centroid 16pp). Penalty giúp PSR_full lên 81.15% nhưng vẫn không bằng Centroid_only.
3. **Subspace_only = 0.54% ≈ random**: Subspace projection một mình hoàn toàn không phân biệt được task.

**Trong whitened space**, cả hai đều thất bại (Centroid_only=4.80%, PSR_full=15.25%) — nhưng lưu ý: đây là PSR Centroid formula, **không phải L2 centroid** của Phase B (L2 centroid whitened = 99.99%). PSR dùng formula khác để tính centroid score.

**Implication**: PSR không phải là "centroid + subspace bonus" — subspace term thực ra là **noise** trong raw space, và penalty là cơ chế vá lỗi. Thiết kế PSR cần xem xét lại.

---

## D-I2 — D2 Rank sweep: PSR accuracy GIẢM khi tăng k (ngược với kỳ vọng)

**Dữ liệu D2 — PSR accuracy theo số chiều subspace k:**

| k | Raw acc | Memory/task | Whitened acc |
|---|---------|-------------|--------------|
| 2 | **87.46%** | 12.3 KB | 51.51% |
| 4 | 84.34% | 20.5 KB | 30.61% |
| **8** | 81.15% | 36.9 KB | 15.25% |
| 16 | 76.84% | 69.7 KB | 7.49% |
| 32 | 73.15% | 135.3 KB | 4.16% |
| 64 | 69.30% | 266.5 KB | 3.89% |

**Hiện tượng đáng chú ý**: Trong raw space, accuracy PSR **giảm đều** khi k tăng: k=2 (87.46%) → k=64 (69.30%). Trong whitened space, cũng giảm mạnh: k=2 (51.51%) → k=64 (3.89%).

**Giải thích**: Khi k tăng, subspace chiếm nhiều chiều hơn và bắt đầu bao phủ cả các hướng nhiễu (noise directions), không chỉ signal. Với k=2, PSR chỉ dùng 2 eigenvector quan trọng nhất — đây là signal mạnh nhất của transformer representation. Tăng k đưa thêm "noise" vào score và làm hỏng routing.

**Optimal k = 2** cho PSR, không phải k=8 như thiết kế ban đầu. Bộ nhớ k=2 (12.3 KB/task) cũng tiết kiệm hơn 3× so với k=8.

**Trong whitened space**: Ngay cả k=2 (51.51%) cũng thất bại — xác nhận lại PSR không tương thích với whitening ở bất kỳ rank nào.

---

## D-I3 — D3 Domain breakdown: PSR yếu nhất ở sentiment domain (raw)

**Dữ liệu D3 — PSR_full accuracy theo domain (raw):**

| Domain | Mean acc | Tasks | Per-task |
|--------|----------|-------|----------|
| misc | 94.34% | 3 | copa=100%, wic=94.7%, qqp=88.3% |
| topic | 90.52% | 3 | agnews=97.4%, yahoo=88.9%, dbpedia=85.3% |
| RC | 81.29% | 2 | multirc=83.8%, boolq=78.7% |
| NLI | 79.47% | 3 | mnli=80.8%, rte=79.1%, cb=78.6% |
| **sentiment** | **71.88%** | 4 | sst2=81.0%, amazon=79.3%, imdb=79.1%, **yelp=48.1%** |

**Nhận xét**:
- **Sentiment domain là điểm yếu nhất** (71.88%) do yelp=48.1% kéo mean xuống. Đây nhất quán với I3 (Phase A): yelp-amazon-imdb có subspace gần nhau, PSR trong raw space bị nhầm lẫn giữa chúng.
- **yelp = 48.1% ≈ random cho 3 sentiment tasks** (1/3=33%, ngưỡng random khi nhầm với amazon/imdb).
- **misc domain cao nhất** (94.34%) do copa/wic có embedding đặc trưng rõ ràng.
- **topic domain** (90.52%): dbpedia thấp nhất (85.3%) trong domain vì 14 classes khó phân biệt.

**Trong whitened space**, sentiment domain chỉ còn 9.48%, NLI = 4.78%, RC = 3.50% — hoàn toàn sụp đổ khi whitening.

---

## D-I4 — D6 CRITICAL: RLS_incremental ≡ RLS_batch (Woodbury update numerically exact)

**Dữ liệu D6 — Incremental simulation:**

| Step | PSR (raw) | RLS_batch (raw) | RLS_inc (raw) | PSR (white) | RLS_batch (white) | RLS_inc (white) |
|------|-----------|-----------------|---------------|-------------|-------------------|-----------------|
| 1 | 100% | 100% | 100% | 100% | 100% | 100% |
| 2 | 99.52% | 100% | 100% | 88.35% | 100% | 100% |
| 3 | 95.07% | 100% | 100% | 48.96% | 100% | 100% |
| 5 | 95.06% | 100% | 100% | 48.60% | 100% | 100% |
| 8 | 91.08% | 100% | 100% | 20.64% | 100% | 100% |
| 10 | 90.09% | 99.998% | 99.998% | 16.02% | 99.998% | 99.998% |
| 15 | **81.15%** | **99.99%** | **99.99%** | **15.25%** | **99.97%** | **99.97%** |

**Insight 1 — RLS_incremental = RLS_batch hoàn toàn**: Tất cả các bước từ 1→15, RLS_incremental cho kết quả **tương đương hoàn toàn** với RLS_batch (sai lệch < 1e-4). Đây là xác nhận quan trọng rằng **Woodbury incremental update không mất precision** — có thể dùng RLS incremental thay thế hoàn toàn cho batch training trong CL setting.

**Insight 2 — PSR degradation theo số task**: PSR giảm từ 100% (1 task) xuống 81.15% (15 tasks) trong raw space. Đây là degradation 18.85pp qua 15 tasks — **average khoảng 1.35pp/task thêm vào**. Nguyên nhân: khi thêm task mới, subspace của các task tương đồng (sentiment) bắt đầu conflict.

**Insight 3 — PSR whitened sụp đổ ngay từ task thứ 3**: PSR_whitened: step 2 = 88.35% (vẫn OK), step 3 = 48.96% (sụp đổ ngay khi thêm task thứ 3). Điều này cho thấy khi có ≥3 task trong whitened space, subspace-based scoring của PSR hoàn toàn không còn tác dụng.

**Insight 4 — RLS robust qua toàn bộ 15 tasks**: RLS (cả batch và incremental) giữ >99.97% accuracy ở cả 15 tasks, cả raw và whitened. **RLS là router tốt nhất cho incremental CL**.

---

## D-I5 — So sánh tổng hợp: PSR vs RLS trong CL setting

| Tiêu chí | PSR (raw) | RLS_incremental (raw) | RLS_incremental (whitened) |
|----------|-----------|----------------------|---------------------------|
| Final acc (15 tasks) | 81.15% | **99.99%** | **99.97%** |
| Degradation 1→15 tasks | −18.85pp | −0.01pp | −0.03pp |
| Incremental update | O(d·k) (SVD update) | O(d²) (Woodbury) | O(d²) (Woodbury) |
| Whitening compatible | ❌ | ✅ | ✅ |
| Memory/task (k=8) | 36.9 KB | ~8 MB (d×d covariance) | ~8 MB |
| Exact incremental | ❌ (SVD approx) | ✅ (Woodbury exact) | ✅ (Woodbury exact) |

**Kết luận**: Trong CL incremental setting, **RLS_incremental là lựa chọn tối ưu** — accuracy 99.99%, không degradation, Woodbury update exact. Chi phí là bộ nhớ O(d²) per task (~8MB cho d=1024), có thể chấp nhận trong hầu hết setting. PSR chỉ có lợi thế bộ nhớ nhỏ (36.9 KB vs 8MB) nhưng đánh đổi 18pp accuracy.

---

## D-I6 — Cần chạy thêm thí nghiệm Phase D nào không?

**Đã đủ:**
- D1 (component ablation): rõ ràng, centroid > PSR_full
- D2 (rank sweep): optimal k=2 cho PSR
- D3 (domain): sentiment là điểm yếu, yelp=48%
- D6 (incremental): RLS_inc ≡ RLS_batch, Woodbury exact

**Có thể bổ sung (low priority):**
- **D2 với k=1**: Thử k=1 (chỉ top eigenvector) để xem có tốt hơn k=2 không trong raw space
- **D6 với L2 centroid + Woodbury whitening incremental**: Confirm L2+whitening fully incremental cũng đạt ~99.99% qua 15 steps
- **D3 cross-backbone**: Domain pattern có giữ nguyên trên T5-base hoặc LLaMA không?

**Không cần:**
- Thêm ablation PSR variants — đã đủ kết luận
- Thêm rank sweep với whitening — PSR whitened thất bại ở mọi k

---

## D-I7 — Summary: 5 insight đáng giá nhất từ Phase D

1. **Centroid_only > PSR_full trong raw space** (82.92% vs 81.15%): Subspace term của PSR là noise, không phải signal. Penalty term là cơ chế vá lỗi, không phải thiết kế chủ động.

2. **Optimal PSR rank k=2, không phải k=8**: Tăng k làm giảm accuracy trong raw space. k=2 tiết kiệm 3× bộ nhớ và cho accuracy cao hơn.

3. **RLS_incremental (Woodbury) = RLS_batch (exact)**: Không có precision loss. Woodbury update hoàn toàn thay thế được batch RidgeClassifier trong CL.

4. **RLS robust qua 15 tasks (99.99%), PSR degradation 18.85pp**: RLS là lựa chọn duy nhất cho production CL router.

5. **PSR sụp đổ từ task thứ 3 trong whitened space**: Không thể dùng PSR trong bất kỳ incremental whitened setting nào.

---

---

# Phase E Theory Validation — Insights

**Backbone**: flan-t5-large (d_model = 1024)  
**Benchmark**: Long_Sequence (15 tasks)  
**Hai runs**: raw embeddings vs. ZCA-whitened embeddings  
**Files**: `theory_flan-t5-large_Long_Sequence.json` / `_whitened.json`  
**Sections**: E1 (KL confusion–accuracy correlation), E2 (Grassmann capacity bound), E3 (Random Matrix Theory signal analysis), E3_shrinkage (OAS covariance shrinkage)

---

## E-I1 — E1: KL confusion có tương quan chặt với PSR accuracy (lý thuyết được xác nhận)

**Dữ liệu E1 — Spearman correlation giữa KL divergence và PSR routing error:**

| Space | Spearman ρ | p-value | PSR acc |
|-------|-----------|---------|---------|
| Raw | **−0.456** | 3.4×10⁻¹² | 82.87% |
| Whitened | **−0.493** | 2.95×10⁻¹⁴ | 12.76% |

**Giải thích**: ρ âm có nghĩa: KL divergence giữa hai task càng lớn (phân phối càng khác nhau) → PSR routing accuracy càng cao. Ngược lại, hai task có phân phối gần nhau → PSR dễ nhầm lẫn hơn.

**Kết luận quan trọng**:
1. Tương quan có ý nghĩa thống kê cực mạnh (p < 10⁻¹¹ cả hai space) → **KL confusion là predictor lý thuyết tốt cho PSR routing difficulty**.
2. ρ ≈ −0.46/−0.49 (không phải −1.0) → KL divergence giải thích được ~21–24% variance của routing accuracy. Còn lại là do các yếu tố hình học khác (subspace overlap, centroid distance).
3. Tương quan mạnh hơn sau whitening (−0.493 > −0.456) dù PSR accuracy rất thấp — điều này có nghĩa rằng **trong whitened space, PSR error pattern nhất quán hơn với KL confusion** (dù overall PSR tệ hơn).
4. **Implication cho paper**: Kết quả E1 là empirical validation của lý thuyết PSR. Có thể trích dẫn p < 10⁻¹² như bằng chứng rằng routing difficulty là predictable từ phân phối dữ liệu.

---

## E-I2 — E2: Grassmann capacity bound — paradox quan trọng

**Dữ liệu E2 — Grassmann manifold capacity analysis:**

| Metric | Raw | Whitened |
|--------|-----|----------|
| T_actual | 15 | 15 |
| T_max_bound | **−45.11** | **187.22** |
| delta_max (max subspace overlap) | 3.84 | 0.316 |
| delta_mean (mean subspace overlap) | 1.39 | 0.032 |
| mean_geodesic_nn | 2.91 | 4.156 |
| **bound_satisfied** | ❌ **false** | ✅ **true** |

**Phân tích:**

**Raw space — bound bị vi phạm (T_max = −45)**: Giá trị âm cho thấy delta (subspace overlap) quá lớn, công thức capacity cho kết quả âm — hệ thống **không thể** đảm bảo phân biệt ngay cả k=8 subspaces. delta_max = 3.84 (từ I3 Phase A: yelp-amazon frobenius_overlap = 3.84) là cực lớn. Tuy nhiên **PSR vẫn đạt 82.87%** — điều này cho thấy PSR hoạt động nhờ mean/centroid signal, không phải nhờ subspace geometry.

**Whitened space — bound thỏa mãn (T_max = 187)**: Sau ZCA whitening:
- delta_mean = **0.032** ≈ 0 (subspace gần như trực giao hoàn toàn, khớp với I3)
- Capacity: có thể chứa **187 tasks** trên Grassmann manifold trước khi bị confuse
- 15 tasks hiện tại chỉ dùng 15/187 = **8% capacity** → còn rất nhiều "chỗ" cho routing

**Paradox cốt lõi (E2 vs Phase B)**:
- Whitened: Grassmann bound thỏa mãn (✅) + capacity 187 tasks... nhưng PSR = 12.76% ❌
- Raw: Grassmann bound vi phạm (❌) + capacity âm... nhưng PSR = 82.87% ✅

**Giải thích nghịch lý**: PSR không khai thác Grassmann geometry trực tiếp — PSR dùng **eigenvalue weighting** để tính projection score. Trong raw space, eigenvalue spectrum anisotropic (dominant directions) tạo ra signal mạnh dù subspace overlap lớn. Trong whitened space, geometry sạch nhưng eigenvalue spectrum flat → PSR không còn discriminative signal.

**Ý nghĩa thực tiễn**: T_max = 187 sau whitening nghĩa là **nếu dùng đúng router (L2 centroid / RLS)**, whitened space có thể scale đến ~187 tasks với accuracy gần hoàn hảo. Grassmann capacity không phải bottleneck — scoring function mới là vấn đề.

---

## E-I3 — E3 RMT: Số chiều signal thực sự là ~280–400, không phải 8

**Dữ liệu E3 — Random Matrix Theory analysis (raw space):**

| Task | n | n_signal_eigvals | evr_k_signal (k=8) | eigenvalue_inflation_ratio |
|------|---|-----------------|--------------------|-----------------------------|
| multirc | 2000 | **355** | **0.982** | 3300 |
| dbpedia | 14000 | 390 | 0.960 | 1402 |
| yelp | 5000 | 326 | 0.956 | 3185 |
| amazon | 5000 | 323 | 0.940 | 1364 |
| boolq | 2000 | 288 | 0.926 | 1260 |
| mnli | 3000 | 288 | 0.907 | 1137 |
| rte | 2000 | 283 | **0.916** | 1560 |
| wic | 2000 | 291 | **0.910** | 800 |
| cb | 250 | 257* | 1.0* | 3.1×10¹⁰* |
| copa | 400 | 401* | 1.0* | 1.25×10¹⁰* |

*cb, copa: σ²_est ≈ 0 (d > n, rank-deficient) → numerical artifact, không đáng tin.

**Kết luận từ raw space RMT**:

1. **~280–400 eigenvalues vượt MP noise bound** — flan-t5-large có **~300 chiều signal thực sự** (không phải 8). Đây là không gian signal rộng hơn nhiều so với k=8 của PSR.

2. **evr_k_signal ở raw space = 0.91–0.98** → top-8 eigenvectors giải thích 91–98% signal subspace. Dù số chiều signal là ~300, chúng tập trung mạnh vào top-8. Đây là lý do k=8 hợp lý trong raw space — và nhất quán với D-I2 (k=2 còn tốt hơn k=8 vì chỉ cần 2 dominant directions).

3. **eigenvalue_inflation_ratio 800–3300** (raw): Top eigenvalue lớn hơn noise 800–3300×. Đây chính là "anisotropy signal" mà PSR khai thác.

---

## E-I4 — E3 RMT (whitened): Signal bị phân tán, k=8 không còn đủ

**Dữ liệu E3 — whitened space:**

| Task | n | n_signal_eigvals | evr_k_signal (k=8) | eigenvalue_inflation_ratio |
|------|---|-----------------|--------------------|-----------------------------|
| yahoo | 10000 | 227 | **0.466** | 5.09 |
| dbpedia | 14000 | 337 | **0.684** | 6.73 |
| mnli | 3000 | 214 | **0.600** | 26.5 |
| rte | 2000 | 200 | **0.587** | 26.2 |
| amazon | 5000 | 250 | **0.642** | 24.3 |
| qqp | 2000 | 220 | **0.674** | 48.6 |
| boolq | 2000 | 216 | 0.641 | 26.8 |
| multirc | 2000 | 340 | **0.914** | 108.7 |
| sst2 | 2000 | 255 | 0.768 | 63.1 |

**Kết luận quan trọng**:

1. **evr_k_signal giảm mạnh sau whitening**: yahoo (0.98→0.47), dbpedia (0.96→0.68), mnli (0.91→0.60). Top-8 eigenvectors chỉ giải thích **46–91% signal** (trung bình ~68%) thay vì 91–98% ở raw.

2. **Đây là nguyên nhân cơ bản khiến PSR thất bại sau whitening**: ZCA whitening cân bằng eigenvalue spectrum → signal không còn tập trung vào 8 dominant directions nữa mà lan ra ~200–340 dimensions. PSR với k=8 chỉ bắt được 47–91% signal → scoring function bị nhiễu bởi 9–53% signal còn lại bị bỏ qua.

3. **eigenvalue_inflation_ratio giảm 100×**: Raw 800–3300 → Whitened 5–109 (trừ cb/copa anomalies). Không còn "extreme dominant direction" nào. Top eigenvalue chỉ lớn hơn noise ~5–108×, không còn 1000–3000×. → Không có "hook point" cho PSR nữa.

4. **Nghịch lý yahoo**: n=10000 (nhiều nhất), nhưng evr_k_signal chỉ 0.466 sau whitening. Sau ZCA, yahoo (10 classes, nhiều data) trở nên đều nhất — signal phân tán rộng nhất. Điều này nhất quán với I2 (yahoo có participation_ratio cao nhất = 626 sau whitening).

---

## E-I5 — E3 Shrinkage: OAS covariance shrinkage không cứu được PSR

**Dữ liệu:**

| Space | PSR acc (raw cov) | PSR acc (shrinkage) | Improvement |
|-------|------------------|---------------------|-------------|
| Raw | 82.87% | **83.20%** | +0.33pp |
| Whitened | 12.76% | **14.86%** | +2.10pp |

**Kết luận**:

1. OAS shrinkage cải thiện PSR rất ít (+0.33pp raw, +2.10pp whitened). **PSR failure là structural, không phải do estimation noise của covariance**.

2. Nếu vấn đề là covariance estimation (do n nhỏ như cb=250, copa=400), shrinkage sẽ giúp nhiều hơn nhiều. Sự cải thiện nhỏ chứng tỏ nguồn error chính không phải ở đây.

3. **oas_shrinkage_alpha** trong raw space: wic (0.020), yelp (0.002), dbpedia (0.002), copa (0.134), cb (0.085). Task lớn cần rất ít shrinkage; copa/cb cần nhiều nhất do n nhỏ, nhưng dù shrink cũng không cải thiện được gì đáng kể.

4. **Whitened space alpha cao hơn** (0.08–0.37) — sau whitening covariance estimation khó hơn, cần shrinkage nhiều hơn. Nhưng vẫn không đủ để cứu PSR.

---

## E-I6 — Tổng hợp paradox các kết quả Phase E

| | Raw space | Whitened space |
|--|-----------|----------------|
| Grassmann bound | ❌ Violated | ✅ T_max=187 |
| evr_k_signal | ✅ 0.91–0.98 | ❌ 0.47–0.91 |
| PSR accuracy | ✅ 82.87% | ❌ 12.76% |
| KL–accuracy correlation | ✅ ρ=−0.46 | ✅ ρ=−0.49 |
| Shrinkage help | ✗ +0.33pp | ✗ +2.10pp |

**Chuỗi nhân quả**: Whitening cân bằng eigenspectrum (evr_k_signal giảm) → k=8 không capture đủ signal → PSR score bị nhiễu → accuracy sụp đổ. Đồng thời: whitening tách các subspace ra (Grassmann bound thỏa mãn) → nhưng PSR không khai thác được sự tách biệt này vì formula sai.

---

## E-I7 — Cần chạy thêm thí nghiệm Phase E nào không?

**Đã đủ:**
- E1 (KL correlation): rõ ràng, ρ=−0.46/−0.49, p<10⁻¹¹ — đủ để báo cáo
- E2 (Grassmann bound): T_max=187 (whitened) vs −45 (raw) — paradox đã được giải thích
- E3 (RMT): n_signal_eigvals ~280–400, evr_k_signal pattern rõ ràng
- E3_shrinkage: không đủ cải thiện để quan tâm

**Có thể bổ sung (medium priority):**
- **E2 với k=2**: Tính lại T_max_bound với k=2 thay vì k=8 — vì D-I2 cho thấy k=2 optimal. T_max cho k=2 trong raw space có thể dương và > 15.
- **E3 với whitening + k=optimal**: Tính evr_k_signal cần bao nhiêu k để đạt 95% signal trong whitened space — có thể k cần ~50–100 thay vì 8.
- **E1 với L2 centroid** thay vì PSR: Kiểm tra xem L2 centroid accuracy có correlate với KL divergence không, hay metric khác phù hợp hơn.

**Không cần:**
- Thêm shrinkage variants — không có impact đáng kể
- Chạy lại E3 với các RMT estimator khác (Ledoit-Wolf, etc.) — OAS đã là state-of-the-art

---

## E-I8 — Summary: 5 insight đáng giá nhất từ Phase E

1. **KL confusion → PSR accuracy: ρ=−0.46, p<10⁻¹²** (raw) — lý thuyết xác nhận thực nghiệm. Tasks càng khác nhau về phân phối → PSR routing càng tốt. Kết quả có thể trích dẫn trực tiếp trong paper.

2. **Grassmann capacity T_max = 187 sau whitening** (vs −45 raw): Whitened space có đủ geometric capacity cho ~12× số tasks hiện tại. PSR thất bại không phải vì thiếu geometric separation mà vì scoring function sai.

3. **~300 signal dimensions thực sự** (không phải 8): RMT cho thấy ~280–400 eigenvalues vượt Marchenko-Pastur bound. PSR chỉ dùng k=8 = ~2–3% signal dimensions.

4. **evr_k_signal sụp đổ sau whitening** (0.47–0.91 vs 0.91–0.98 raw): Đây là cơ chế cơ bản giải thích PSR failure — ZCA phân tán signal ra nhiều dimensions, k=8 không còn đủ để capture discriminative subspace.

5. **OAS shrinkage không cứu được PSR** (+0.33pp raw): Vấn đề PSR là thiết kế (formula), không phải covariance estimation. Không có "quick fix" nào cho PSR trong whitened space.
