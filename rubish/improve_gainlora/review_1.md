# Sàng lọc & Đánh giá Ý tưởng Cross-Domain cho SpecRoute V7

> **Mục đích**: Đánh giá 3 hướng tiềm năng nhất từ cross_domain_ideas.md, qua 3 vòng phân tích + phản biện độc lập.
> **Ràng buộc**: zero-replay nghiêm ngặt (settings.txt), theory-first (work_ethic.txt)

---

## I. Bối cảnh vấn đề

### Vấn đề cốt lõi (từ V6)

V6 (SVD routing + C4, không prototypes) đạt AP(EM) ≈ 27.4 — thất bại. Nguyên nhân gốc: **Null-space collapse**.

**Cơ chế**: GPM tích lũy basis từ activation covariance qua mỗi task. Null-space thu hẹp dần:
- Layer 7: 8/512 → 161/512 (task 8, IMDB) → 344/512 (task 13)
- Các hướng còn lại trong null-space KHÔNG ĐẢM BẢO alignment với feature space của task mới
- Kết quả: IMDB eval_loss = 7.35 → 6.37 (10 epochs), EM = 0.0 suốt — LoRA-B **không thể học**

**Ba khía cạnh của vấn đề**:
1. **Tiêu thụ quá nhiều**: Mỗi task tiêu thụ ~26 dims trung bình (Layer 7), tổng 344/512 sau 13 tasks
2. **Tiêu thụ dư thừa**: Same-domain tasks (yelp/amazon, agnews/yahoo) có activation tương tự → GPM bases trùng lặp → lãng phí capacity
3. **Sử dụng kém hiệu quả**: Null-space init bằng random Kaiming → hướng A_t ngẫu nhiên trong null-space → có thể bỏ lỡ hướng task-relevant

### Ràng buộc thiết kế

| Ràng buộc | Nguồn | Hệ quả |
|-----------|-------|--------|
| Zero-replay | settings.txt | Không prototype, không data statistics tại inference |
| Theory-first | work_ethic.txt | Mọi thay đổi phải có lập luận toán học/lý thuyết thông tin |
| Giữ narrative | SPECROUTE_IDEA.md | Tốt nhất là bảo tồn Routing-Protection Duality |
| Chỉ sửa khi cần | working_method.txt | Nếu idea hiện tại đã chặt chẽ, chỉ sửa implementation |

---

## II. Top 3 Hướng Tiềm Năng

| # | Hướng | Nguồn cross-domain | Khía cạnh giải quyết |
|---|-------|--------------------|--------------------|
| 1 | **Selective Spectral Protection** | Synaptic Tagging & Capture (Neuroscience) + Information Geometry | Tiêu thụ quá nhiều |
| 2 | **Subspace Defragmentation** | Memory Defragmentation (OS) + NESS (Signal Processing) | Tiêu thụ dư thừa |
| 3 | **Data-Informed Subspace Tracking** | PAST/CoSO (Signal Processing) + Grassmannian Optimization | Sử dụng kém hiệu quả |

**Tại sao 3 hướng này?**
- Ba hướng giải quyết 3 khía cạnh KHÁC NHAU của null-space collapse
- Cả 3 đều zero-replay compliant (chỉ dùng model artifacts, không data)
- Cả 3 đều tương thích với Routing-Protection Duality framework hiện tại
- Tiềm năng kết hợp: $\text{Direction 1} \circ \text{Direction 2} \circ \text{Direction 3}$ tạo framework thống nhất

**Tại sao loại bỏ các hướng khác?**
- *Compressed Sensing*: Task identity không thực sự sparse — mỗi input thuộc đúng 1 task nhưng activation phân bố trên toàn bộ d dims
- *Hypernetworks*: Mâu thuẫn với SpecRoute narrative (không learned routing), thêm complexity lớn
- *Neural ODEs/SDEs*: Quá trừu tượng, khó ánh xạ cụ thể vào GPM mechanism
- *Cellular Automata*: Dynamic rank thú vị nhưng không giải quyết trực tiếp null-space collapse
- *TDA*: Persistence diagrams khó ánh xạ thành actionable change cho GPM

---

## III. Round 1 — Selective Spectral Protection (Bảo vệ Phổ Chọn lọc)

### A. Phân tích chuyên sâu

#### Cảm hứng: Synaptic Tagging & Capture (STC)

Trong não bộ (Frey & Morris, 1997, Nature):
1. Khi synapse được kích hoạt mạnh → tạo **Tag** hoá học tại synapse
2. Neuron tiết **Plasticity-Related Proteins (PRP)**
3. Chỉ synapse có Tag mới **Capture** được PRP → chỉ synapse quan trọng được ổn định hoá

**Ánh xạ sang SpecRoute**:
- Synapse ← GPM basis direction (hướng trong không gian activation)
- Kích hoạt mạnh ← Singular value $\sigma_i$ lớn (spectral importance cao)
- Tag ← $\text{imp}(i) = \sigma_i^2 / \sum_j \sigma_j^2 > \gamma$ (vượt ngưỡng importance)
- Capture/PRP ← Đưa vào GPM basis (bảo vệ vĩnh viễn)
- **Kết quả**: Chỉ bảo vệ hướng QUAN TRỌNG, giải phóng hướng không quan trọng cho task sau

#### Phát biểu toán học

**Hiện tại** (GPM gốc): Sau training task $t$, tính activation covariance $C_t = \mathbb{E}[a_t a_t^T]$, SVD → giữ tất cả eigenvectors có eigenvalue tích luỹ < threshold.

**Đề xuất**: Thêm bước **lọc chọn lọc** dựa trên spectral importance của expert đã trained.

Sau khi train task $t$:
1. Tính spectral signature: $B_t A_t = U_t \Sigma_t V_t^T$
2. Với mỗi GPM basis $b_i$ (từ activation covariance), tính:
$$\text{relevance}(b_i) = \sum_{j=1}^{r} \sigma_{t,j}^2 \cdot (v_{t,j}^T b_i)^2$$
3. Chỉ giữ GPM basis $b_i$ nếu $\text{relevance}(b_i) > \gamma \cdot \max_k \text{relevance}(b_k)$

**Ý nghĩa**: relevance(b_i) đo lường mức độ mà hướng $b_i$ trong activation space ảnh hưởng đến expert output. Hướng có relevance thấp = hướng mà dữ liệu đi qua NHƯNG expert KHÔNG phản ứng → **không cần bảo vệ**.

#### Phân tích tác động lên capacity

**Hiện tại V6** (13 tasks, Layer 7):
- Tổng GPM dims: 344/512 (67% consumed)
- Null-space còn lại: 168 dims

**Với selective protection** ($\gamma = 0.5$, giả sử giữ ~60% GPM bases):
- Tổng GPM dims ước tính: ~206/512 (40% consumed)
- Null-space: ~306 dims (+82% so với hiện tại)

**Tại IMDB (task 8)**:
- Hiện tại: 161/512 consumed → 351 free
- Selective: ~97/512 consumed → 415 free
- Thêm 64 dims null-space → xác suất chứa IMDB-relevant directions tăng đáng kể

#### Tương thích với Routing-Protection Duality

**Định lý 1** (Routing margin): $\alpha_{t^*}(h) - \max_{t \neq t^*} \alpha_t(h) \geq \kappa_{\min}(t^*) - \varepsilon \cdot \kappa_{\max}$

Selective protection tăng $\varepsilon$ cho weak directions nhưng **không ảnh hưởng routing** vì:
- Routing dùng $\sigma^2$-weighted affinity
- Hướng bị bỏ bảo vệ có $\sigma$ nhỏ → đóng góp nhỏ vào routing score
- Kể cả task mới overlap trên hướng weak này, routing score của task cũ hầu như không giảm

**Chính thức**: Nếu bỏ bảo vệ hướng $v_{t,i}$ với $\sigma_{t,i}^2 < \gamma \cdot \sum_j \sigma_{t,j}^2$:
$$\Delta \alpha_t(h) = \frac{\sigma_{t,i}^2 (v_{t,i}^T h)^2}{\sum_j \sigma_{t,j}^2 \|h\|^2} \leq \gamma$$

Routing margin giảm tối đa $\gamma$. Với $\gamma = 0.05$ (5%), margin giảm tối đa 0.05 — chấp nhận được.

#### Tham chiếu mở rộng

- **EWC** (Kirkpatrick et al., 2017): Dùng Fisher Information Matrix để đánh trọng số importance cho từng parameter. Selective Protection là phiên bản EWC ở mức **subspace** thay vì parameter.
- **PackNet** (Mallya & Lazebnik, 2018): Prune weights không quan trọng sau mỗi task → giải phóng capacity. Tương tự nhưng ở subspace level.
- **TRGP** (Lin et al., 2022): Trust Region GPM — cho phép overlap có kiểm soát dựa trên gradient trust region.
- **NISPA** (Gurbuz & Dovrolis, 2022): Neuroinspired Stability-Plasticity Adaptation — selective freeze dựa trên synapse importance.

#### Zero-replay compliance

GPM bases = model behavior artifact (eigenvectors của activation covariance, tính trong training). Spectral importance $\sigma_i$ = model output artifact (tính từ frozen weights $B_t, A_t$). **Cả hai không phải data statistics** → ✅ zero-replay compliant.

#### Điểm mạnh

1. **Trực tiếp giải quyết root cause**: Giảm GPM consumption rate
2. **Tự nhất quán**: Importance cho protection = importance cho routing (cả hai dùng $\sigma^2$)
3. **Tăng cường Duality**: Protection → routing trở thành protection *importance-weighted* → routing *importance-weighted*
4. **Dễ implement**: Chỉ thêm bước lọc sau `get_repsentation()`, trước khi save `reg_{}.pt`
5. **Hyperparameter rõ ràng**: $\gamma$ có ý nghĩa information-theoretic (fraction of spectral energy not protected)

---

### B. Phản biện Độc lập — Round 1

*[Đổi vai: reviewer khách quan, khó tính]*

#### Phản biện 1: GPM bases ≠ Spectral directions — mismatch cơ bản

**Vấn đề nghiêm trọng**: Phân tích trên NHẦM LẪN hai khái niệm:
- **GPM bases** ($b_i$): Eigenvectors của activation covariance $C_t = \mathbb{E}[a_t a_t^T]$, tính SAU khi InfLoRA project A vào null-space. Đại diện cho các hướng mà **data đi qua** lora_A.
- **Spectral signatures** ($v_{t,j}$, $\sigma_{t,j}$): SVD của $B_t A_t$, tính SAU khi train xong. Đại diện cho các hướng mà **expert đã học** để biến đổi input.

Hai tập này CHỒNG CHÉO nhưng KHÔNG TRÙNG KHỚP:
- GPM bases span $\text{rowspan}(A_t)$ (qua activation covariance)
- $V_t$ rows cũng nằm trong $\text{rowspan}(A_t)$ (từ SVD)
- Nhưng **thứ tự importance** có thể khác: GPM eigenvalue ∝ **data variance** dọc hướng đó; $\sigma^2$ ∝ **expert modification gain**

Hướng có data variance CAO nhưng modification gain THẤP = hướng mà data biến động NHIỀU nhưng expert CHỌN KHÔNG biến đổi mạnh. Selective protection theo $\sigma$ sẽ bỏ bảo vệ hướng này. Nhưng nếu hướng này QUAN TRỌNG cho task khác?

**Mức độ nghiêm trọng**: TRUNG BÌNH. Có thể giảm thiểu bằng cách dùng relevance metric kết hợp cả hai: $\text{score}(b_i) = \lambda \cdot \text{eigenvalue}(b_i) + (1-\lambda) \cdot \text{spectral\_relevance}(b_i)$.

#### Phản biện 2: Ngưỡng $\gamma$ nhạy cảm và task-dependent

Số lượng hướng "quan trọng" khác nhau theo task:
- Task đơn giản (sentiment classification): có thể chỉ cần 1-2 hướng chính ($\sigma_1 \gg \sigma_2$)
- Task phức tạp (NLI, multi-class): cần nhiều hướng ($\sigma$ phân bố đều)

$\gamma$ cố định sẽ:
- Quá aggressive cho task phức tạp → mất bảo vệ cần thiết
- Quá conservative cho task đơn giản → tiết kiệm ít capacity

**Mức độ nghiêm trọng**: THẤP-TRUNG BÌNH. Có thể dùng adaptive $\gamma$ theo spectral entropy: task có entropy cao (nhiều hướng quan trọng) → $\gamma$ thấp hơn (giữ nhiều hơn).

#### Phản biện 3: Tác động thực tế bị overestimate

Phân tích ước tính "giữ ~60% GPM bases" — nhưng con số này cần kiểm chứng empirically:
- Nếu spectral energy tập trung (top-2 chiếm 90%): tiết kiệm đáng kể (~60-70% dims)
- Nếu spectral energy phân bố đều (C4 entropy regularization khuyến khích điều này!): tiết kiệm ít (~10-20% dims)

**Irony**: C4 (spectral entropy regularization) KHUYẾN KHÍCH LoRA sử dụng full rank → spectral energy phân bố đều → SELECTIVE PROTECTION ÍT HIỆU QUẢ HƠN!

Đây là **mâu thuẫn nội tại**: C4 làm expert tốt hơn bằng cách spread energy → nhưng selective protection cần energy tập trung để tiết kiệm capacity.

**Mức độ nghiêm trọng**: CAO. Đây là mâu thuẫn cấu trúc. Cần chọn: (a) bỏ C4 entropy reg, hoặc (b) giảm hiệu quả selective protection, hoặc (c) tìm cách dung hoà.

**Phương án dung hoà khả thi**: Entropy reg khuyến khích uniform σ TRONG r dims. Nhưng ta có thể chọn KHÔNG bảo vệ toàn bộ r dims mà chỉ bảo vệ top-k (k < r) CÒN LẠI selective. Kể cả khi σ uniform, nếu k=4 thay vì r=8, ta vẫn tiết kiệm 50% GPM dims.

#### Phản biện 4: Không giải quyết routing failure

Selective protection chỉ giải quyết **null-space collapse** (vấn đề 1: never-learning). Nhưng V6 cũng có **routing degradation** (vấn đề 2: yelp 55→36) — hướng này KHÔNG giải quyết misrouting.

Nếu null-space rộng hơn, expert mới học tốt hơn, nhưng SVD routing vẫn có thể phân bổ weight SAI vì assumption $h \in \text{span}(V_{t^*})$ vẫn không đúng trong thực tế.

**Mức độ nghiêm trọng**: TRUNG BÌNH. Selective protection là điều kiện CẦN (expert phải học được) nhưng chưa ĐỦ (routing cũng phải đúng). Cần kết hợp với hướng khác.

#### Kết luận Round 1

**Đánh giá tổng thể**: Selective Spectral Protection là hướng **HỮU ÍCH nhưng KHÔNG ĐỦ**.
- ✅ Giảm GPM consumption → more null-space → later tasks HỌC ĐƯỢC
- ❌ Không fix routing failure
- ⚠️ Mâu thuẫn với C4 entropy regularization
- ⚠️ GPM bases vs spectral directions mismatch cần xử lý

**Khuyến nghị**: Giữ làm MỘT PHẦN của giải pháp V7, nhưng cần kết hợp với hướng khác cho routing. Cần giải quyết mâu thuẫn C4.

---

## IV. Round 2 — Subspace Defragmentation (Chống phân mảnh Không gian con)

### A. Phân tích chuyên sâu

#### Cảm hứng: Memory Defragmentation (OS)

Trong hệ điều hành:
- Processes allocate/deallocate memory blocks → **fragmentation**: tổng bộ nhớ trống ĐỦ nhưng bị PHÂN MẢNH → không đủ block liên tục cho process mới
- **Defragmentation**: Dồn các block đã allocate lại → tạo vùng trống liên tục lớn

Trong SpecRoute:
- Tasks "allocate" GPM dims → same-domain tasks tạo bases gần tương tự → **dư thừa** (redundancy ≈ fragmentation)
- Yelp (sentiment, task 1) và Amazon (sentiment, task 2) có activation covariance tương tự → GPM bases overlap
- **Defragmentation**: SVD toàn bộ GPM bases → nén lại loại bỏ dư thừa → giải phóng capacity

#### Phát biểu toán học

Gọi $\mathcal{B} = [\mathcal{B}_1 | \mathcal{B}_2 | \ldots | \mathcal{B}_T] \in \mathbb{R}^{d \times N}$ là ma trận ghép tất cả GPM bases ($N = \sum_t n_t$ với $n_t$ là số dims task $t$).

Compute SVD: $\mathcal{B} = U_{\mathcal{B}} \Sigma_{\mathcal{B}} W^T$

**Key observation**: Nếu task $i$ và task $j$ share activation directions, $\mathcal{B}_i$ và $\mathcal{B}_j$ có overlap → rank($\mathcal{B}$) < $N$. Các singular values nhỏ của $\mathcal{B}$ đại diện cho sự DƯ THỪA.

**Defragmented basis**: $\hat{\mathcal{B}} = U_{\mathcal{B}}[:, 1:k]$ với $k$ chọn sao cho:
$$\frac{\sum_{i=1}^{k} \sigma_{\mathcal{B},i}^2}{\sum_{j=1}^{N} \sigma_{\mathcal{B},j}^2} \geq \rho \quad (\text{e.g., } \rho = 0.99)$$

**Tiết kiệm capacity**: $k < N$ khi các task share directions → null-space tăng $N - k$ dims.

#### Kết nối với NESS (Null-space from Small Singular values)

NESS (Kong et al., 2022) đề xuất: thay vì tìm null-space chính xác (orthogonal complement), dùng **small singular values** để ước lượng "information null-space" — hướng mà tổng energy từ TẤT CẢ experts rất thấp.

Sau defragmentation, các hướng tương ứng singular values NHỎ NHẤT của $\mathcal{B}$ chính là "information null-space" — hướng AN TOÀN NHẤT cho task mới sử dụng:
$$\text{Safe directions} = U_{\mathcal{B}}[:, k+1:d]$$

Đây KHÔNG CHỈ là null-space toán học mà là **null-space thông tin** — có thể RỘNG HƠN null-space toán học nếu ta chấp nhận ε overlap.

#### Tác động lên vấn đề gốc

**Ước tính redundancy trong V6 (15-task sequence Long Order 3)**:
- yelp + amazon: sentiment tasks → similarity cao trong activation space
- agnews + yahoo + dbpedia: topic classification → overlap vừa
- IMDB + SST2: cũng sentiment → overlap với yelp/amazon
- Ước tính: ~15-25% GPM dims là redundant

**Layer 7**: 344 dims → defragmented ~260-290 dims → tiết kiệm 54-84 dims → null-space tăng từ 168 → 222-252 dims (+32-50%)

#### V6 scenario (IMDB task 8 defragmented):

Tại thời điểm bắt đầu train IMDB:
- Hiện tại: 161/512 consumed → 351 free
- Defragmented (sau 7 tasks yelp→rte): ước tính 120-135/512 consumed → 377-392 free
- Tăng ~26-41 dims → cải thiện nhưng IMDB vẫn có thể fail nếu task-relevant directions vẫn nằm ngoài null-space

#### Kết hợp với Direction 1 (Selective Protection)

**Selective + Defragmented**:
1. Selective: giữ top-k important dims mỗi task → N_selected < N_total
2. Defragment: nén N_selected loại bỏ cross-task redundancy → k_compact < N_selected

**Pipeline**: Train task t → Compute spectral importance → Selective filter GPM bases → Defragment toàn bộ accumulated bases → Save.

**Tác động tổng hợp**: Nếu selective giữ 60% VÀ defrag nén thêm 20% → chỉ còn 48% dims original → null-space gần GẤP ĐÔI so với hiện tại.

#### Zero-replay compliance

GPM bases = model artifacts. SVD(GPM bases) = mathematical operation trên model artifacts. Defragmented bases = compressed model artifacts. **KHÔNG dùng data nào** → ✅ zero-replay compliant.

#### Tham chiếu mở rộng

- **CoSO** (Wang et al., 2022): Continuous Subspace Optimization — SVD nén gradient subspaces liên tục. Rất gần với defragmentation nhưng họ dùng cho gradient, ta dùng cho GPM bases.
- **NESS** (Kong et al., 2022): Null-space from Small Singular values — dùng σ nhỏ để ước lượng "rác" space. Chúng ta dùng inverse: σ LỚN = protected, σ nhỏ = safe to reuse.
- **Incremental PCA** (Ross et al., 2008): Online update PCA bases khi data mới đến. Conceptually tương tự defragmentation.

#### Tương thích lý thuyết Routing-Protection Duality

Defragmentation BẢO TOÀN protection (với $\rho$ gần 1):
$$\|P_{\text{defrag}} - P_{\text{original}}\|_F^2 \leq 1 - \rho$$

trong đó $P = \mathcal{B}\mathcal{B}^T$ là projection matrix. Với $\rho = 0.99$, sai số bảo vệ ≤ 1% → ảnh hưởng routing margin tối đa 1%.

Nhưng lợi ích capacity có thể tăng 30-50% → trade-off RẤT HỮU LỢI.

#### Kết nối ngược vấn đề gốc + phản biện Round 1

Round 1 chỉ ra rằng Selective Protection **mâu thuẫn với C4 entropy regularization** (C4 spread energy → selective ít hiệu quả). 

Defragmentation KHÔNG gặp mâu thuẫn này vì nó hoạt động ở mức **cross-task redundancy**, không phụ thuộc vào energy distribution TRONG một task. Kể cả mỗi task có σ uniform (do C4), nếu 2 tasks share activation directions, defrag vẫn nén được.

→ Defragmentation BỔ SUNG cho Selective Protection đặc biệt trong trường hợp C4 làm selective ít hiệu quả.

---

### B. Phản biện Độc lập — Round 2

*[Đổi vai: reviewer khách quan, khó tính]*

#### Phản biện 1: Defragmentation thay đổi projection operator — hệ quả tinh vi

Khi defragment $P_{\text{old}} → P_{\text{defrag}}$, tất cả task SAU defrag sẽ dùng $P_{\text{defrag}}$ để tính null-space. Nhưng các task ĐÃ TRAIN dùng $P_{\text{old}}$ → có mismatch:
- $A_t$ được projected theo $P_{\text{old}}$, nhưng bảo vệ theo $P_{\text{defrag}}$
- Nếu $P_{\text{defrag}}$ bỏ đi hướng mà $A_t$ sử dụng → mất bảo vệ cho task $t$
- Task mới có thể overlap vào hướng này → gây interference

**Mức độ nghiêm trọng**: TRUNG BÌNH. Defrag SVD với $\rho = 0.99$ chỉ bỏ hướng có energy < 1% → hướng bị bỏ gần như không quan trọng cho BẤT KỲ task nào. Nhưng cần prove rigorously rằng protection guarantee chỉ giảm $O(1-\rho)$.

#### Phản biện 2: Chia sẻ same-domain activation THỰC SỰ redundant hay cần thiết?

Giả định "yelp và amazon có GPM bases overlap" — nhưng nếu overlap này là DO Ý ĐỒNG (designed feature):
- Hướng [positive_sentiment] được CẢ HAI task dùng → overlap CAO
- Nhưng yelp dùng cho restaurant, amazon cho products — context KHÁC
- Nếu defrag gộp hướng [positive_sentiment] thành 1 vector thay vì 2 → mất resolution giữa yelp-sentiment và amazon-sentiment
- Routing có thể GIẢM chất lượng vì 2 task sentiment trở nên KHÁC BIỆT NHAU ÍT HƠN trong protected space

**Mức độ nghiêm trọng**: THẤP. Routing dùng spectral signatures ($V_t, \sigma_t$ từ $B_t A_t$), KHÔNG phải GPM bases trực tiếp. GPM bases ảnh hưởng GIÁN TIẾP qua null-space projection. Nếu defrag giảm protected dims nhưng spectral signatures vẫn khác nhau (vì $B_t$ khác nhau), routing vẫn OK.

#### Phản biện 3: Lợi ích THỰC SỰ bao nhiêu?

25% redundancy → tiết kiệm ~80 dims → null-space tăng từ 168→248.

Nhưng từ V6 data: IMDB (task 8) có 351 dims free VẪN FAIL. Vậy thêm 40 dims (defrag tại task 8) có thực sự giải quyết vấn đề? 

**Phản lập luận**: 351 dims free nhưng random init A → chỉ một phần nhỏ task-relevant. Defrag + data-informed init (Direction 3) mới khai thác được lợi ích: more space + BETTER utilization of that space.

**Mức độ nghiêm trọng**: TRUNG BÌNH. Defrag một mình CÓ THỂ không đủ → cần kết hợp Direction 3.

#### Phản biện 4: Computational overhead

Mỗi task phải:
1. Ghép tất cả GPM bases: $O(d \times N)$
2. SVD: $O(d \times N^2)$ hoặc $O(d^2 \times N)$ tuỳ kích thước

Với d=512, N tăng dần (tối đa ~344): SVD(512×344) không quá nặng (~0.1s trên GPU). Nhưng nếu scale lên d=4096 (LLama): SVD(4096×2000+) có thể nặng.

**Mức độ nghiêm trọng**: THẤP. Post-training operation, chạy MỘT LẦN per task. Có thể dùng randomized SVD cho scale lớn.

#### Kết luận Round 2

**Đánh giá tổng thể**: Subspace Defragmentation là hướng **BỔ SUNG TỐT** cho Selective Protection.
- ✅ Giải quyết cross-task redundancy (Direction 1 không cover)
- ✅ Không mâu thuẫn C4 entropy reg
- ✅ Zero-replay compliant
- ⚠️ Lợi ích hạn chế nếu dùng RIÊNG (cần Direction 3)
- ⚠️ Protection guarantee cần proof chặt

**Khuyến nghị**: Kết hợp Direction 1 + Direction 2 + Direction 3 thành pipeline thống nhất. Direction 2 mạnh nhất khi dùng cùng Direction 1 (selective → defrag → save) và Direction 3 (defrag → data-informed init).

---

## V. Round 3 — Data-Informed Subspace Tracking (Khởi tạo Có hướng dẫn Dữ liệu)

### A. Phân tích chuyên sâu

#### Cảm hứng: Subspace Tracking & PAST (Signal Processing)

Trong xử lý tín hiệu:
- **PAST** (Projection Approximation Subspace Tracking, Yang 1995): Cập nhật PCA bases khi data mới đến. Thay vì random init, dùng thông tin từ data hiện tại để chọn hướng TỐT NHẤT trong không gian sẵn có.
- **CoSO** (Continuous Subspace Optimization, Wang et al., 2022): Tối ưu hoá subspace liên tục thay vì cố định từ đầu.

#### Ánh xạ sang SpecRoute — vấn đề gốc

Hiện tại InfLoRA:
```
A_random = kaiming_init()  # random directions
A_t = A_random - A_random @ P_old  # project to null-space
A_t = normalize(A_t)
```

Kết quả: $A_t$ có r=8 hàng NGẪU NHIÊN trong null-space. Nếu null-space 351 dims (IMDB), 8 hàng random có xác suất thấp capture đúng hướng IMDB cần.

**Đề xuất**: Dùng data thực để tìm hướng QUAN TRỌNG NHẤT trong null-space:

```
# Phase 1: Warm-up — thu thập activation covariance (vài batch đầu)
C_data = E[h h^T]  # input activation covariance của task mới

# Phase 2: Project covariance eigenvectors vào null-space
U_data, S_data, _ = SVD(C_data)
U_proj = U_data - P_old @ U_data  # project each eigenvector to null-space
# Re-orthogonalize
U_proj_ortho = QR(U_proj).Q

# Phase 3: Chọn top-r directions theo projected eigenvalue
scores = [S_data[i] * ||U_proj[:, i]||^2 for i in range(d)]
top_r = argsort(scores, descending=True)[:r]

# Phase 4: Set A_t
A_t = U_proj_ortho[:, top_r].T  # (r, d) — task-relevant null-space directions
```

#### Phát biểu toán học

**Bài toán tối ưu**: Tìm $A_t \in \mathbb{R}^{r \times d}$ sao cho:
$$\max_{A_t} \quad \text{tr}(A_t \, C_{\text{data}} \, A_t^T) \quad \text{s.t.} \quad A_t P_{\text{old}} = 0, \quad A_t A_t^T = I_r$$

Tức là: maximize variance captured (task-relevance) trong ràng buộc null-space.

**Lời giải**: Đây là bài toán **Constrained PCA** — PCA trên projected data. 

Gọi $Q = I - P_{\text{old}}$ là null-space projector. Ta cần:
$$\max_{A_t} \quad \text{tr}(A_t \, Q C_{\text{data}} Q \, A_t^T) \quad \text{s.t.} \quad A_t A_t^T = I_r$$

Lời giải chính xác: $A_t$ = top-r eigenvectors của $Q C_{\text{data}} Q$ (projected covariance).

**Ý nghĩa information-theoretic**: $A_t$ capture **mutual information** $I(h; \text{task features})$ tối đa trong ràng buộc null-space. Theo Data Processing Inequality:
$$I(A_t h; y) \leq I(h; y)$$

nhưng data-informed $A_t$ MAXIMIZE $I(A_t h; y)$ trong null-space, trong khi random $A_t$ chỉ capture một phần RANDOM.

#### Tại sao đây là thay đổi ĐÁNG KỂ so với hiện tại

**Hiện tại** (random init): $A_t$ rows = random vectors trong null-space. Nếu null-space 351D và task cần 8D, xác suất align tốt:
- Johnson-Lindenstrauss: Random 8D projection bảo toàn khoảng cách với $\epsilon$ error nếu $8 \geq O(\log(n)/\epsilon^2)$
- Nhưng J-L chỉ bảo toàn KHOẢNG CÁCH, không bảo toàn TASK-RELEVANCE
- Ví dụ: IMDB cần capture [positive_sentiment, negative_sentiment, movie_genre, ...] → random 8D trong 351D null-space CÓ THỂ BỎ LỠ hoàn toàn nếu sentiment directions nằm ngoài null-space (đã bị yelp/amazon claim)

**Data-informed**: $A_t$ rows = hướng có variance CAO NHẤT trong null-space theo data IMDB. Nếu sentiment directions bị yelp/amazon claim (ngoài null-space), data-informed sẽ tìm hướng TƯƠNG TỰ GẦN NHẤT trong null-space → **tốt hơn random**.

#### Kết hợp với Direction 1 + 2

**Pipeline thống nhất**:
1. **Selective Protection** (sau train task t-1): Lọc GPM bases theo spectral importance → P_selective
2. **Defragmentation** (sau selective): SVD-compact P_selective → P_compact (nhỏ hơn)
3. **Data-Informed Init** (đầu train task t): Thu thập vài batch data → tính projected covariance → top-r eigenvectors → A_t

**Hiệu ứng cộng dồn**:
- Direction 1: Null-space RỘNG hơn (ít dims bị protect)
- Direction 2: Null-space RỘNG hơn nữa (loại bỏ redundancy)
- Direction 3: A_t CHỌN hướng TỐT NHẤT trong null-space rộng đó

**Dự đoán**: Combined approach → IMDB (task 8) có ~400+ dims null-space VÀ A_t aligned với IMDB features → eval_loss giảm nhanh → EM > 0 → null-space collapse GIẢI QUYẾT.

#### Zero-replay compliance

Data-informed init dùng **training data** của task HIỆN TẠI (đang train). Đây KHÔNG phải replay (replay = dùng lại data CŨ). Training data của task hiện tại luôn available trong CL setting → ✅ zero-replay compliant.

#### Tham chiếu mở rộng

- **PAST** (Yang, 1995): Online subspace tracking algorithm. Ta chỉ dùng concept (data-informed subspace), không cần online aspect vì ta xử lý per-task.
- **Riemannian Optimization on Grassmann Manifold** (Edelman et al., 1998): Tối ưu trên đa tạp Grassmann — A_t initialization là một điểm trên $\text{Gr}(r, \text{null-space-dim})$. Constrained PCA chính là gradient ascent trên Grassmannian.
- **CoSO** (Wang et al., 2022): Dùng SVD để nén gradient subspace → ta dùng SVD để chọn initialization subspace.
- **InfLoRA** (Liang et al., 2024): Bản gốc dùng random init + null-space projection. Data-informed init là natural extension.

#### Lý thuyết phổ Grassmannian

$A_t$ có r hàng trong $\mathbb{R}^d$ → $\text{rowspan}(A_t) \in \text{Gr}(r, d)$. Null-space projection giới hạn ta trên **restricted Grassmannian** $\text{Gr}(r, d - N_{\text{protected}})$.

**Optimal packing trên restricted Grassmannian**: Data-informed init tìm điểm tối ưu cho task performance, trong khi random init lấy MỘT điểm ngẫu nhiên. Với $\text{dim}(\text{Gr}(8, 351)) = 8 \times 343 = 2744$, không gian lựa chọn RẤT LỚN → random gần như chắc chắn sub-optimal.

#### Tương thích Routing-Protection Duality

Data-informed $A_t$ capture task features TỐT HƠN → $B_t$ trained trên relevant subspace → $\sigma_t$ LỚN HƠN → spectral signature MẠNH HƠN → routing margin $\kappa_{\min}(t)$ TĂNG → Theorem 1 bound CHẶT HƠN.

**Đây chính là missing piece**: V6 fail vì expert KHÔNG HỌC ĐƯỢC (σ ≈ 0). Data-informed init trực tiếp tấn công vấn đề này bằng cách đảm bảo A_t aligned với task features → B_t CÓ THỂ HỌC → σ > 0 → routing HOẠT ĐỘNG.

---

### B. Phản biện Độc lập — Round 3

*[Đổi vai: reviewer khách quan, khó tính]*

#### Phản biện 1: Warm-up phase tốn step và gây training delay

Data-informed init cần vài batch data TRƯỚC KHI set A_t (vì phải tính $C_{\text{data}}$). Nhưng InfLoRA cần A_t cố định TRƯỚC khi training bắt đầu (A frozen sau init). 

Hai cách giải quyết:
- (a) Dùng separate warm-up pass (forward-only, vài batch) → tính C_data → set A_t → rồi mới bắt đầu training
- (b) Dùng "lazy init": train vài step với random A, rồi re-init → nhưng điều này phức tạp và có thể gây instability

**Mức độ nghiêm trọng**: THẤP. Option (a) rất tự nhiên — thêm 1 forward pass nhỏ (100-200 batch, ~30 giây) trước training. NHƯNG: phần `get_repsentation()` hiện tại ĐÃ LÀM forward pass qua toàn bộ training data (lên tới 1000 batch)! Data-informed init có thể **tận dụng cùng forward pass này** — chỉ cần collect input covariance ĐỒNG THỜI khi collect activation covariance cho GPM.

**Timing issue**: `get_repsentation()` chạy SAU training (để update GPM). Nhưng ta cần data-informed init TRƯỚC training. 

→ **Giải pháp**: Thêm `pre_task_data_collection()` chạy TRƯỚC training, forward vài batch để tính $C_{\text{data}}$, rồi projected PCA → set A_t. SAU training, `get_repsentation()` chạy bình thường cho GPM. Overhead: 1 thêm forward pass nhỏ per task.

#### Phản biện 2: Constrained PCA có thể cho directions kém nếu null-space quá narrow

Nếu null-space chỉ còn 168 dims (task 13) và task cần 8 dims: projected covariance $QC_{\text{data}}Q$ có 168 eigenvectors. Top-8 CHẮC CHẮN tốt hơn random-8.

Nhưng nếu null-space chỉ còn 20 dims (task rất muộn trong sequence dài): projected covariance gần degenerate → top-8 gần random → lợi ích data-informed GIẢM.

**Mức độ nghiêm trọng**: THẤP cho 15-task sequence (null-space ≥ 168 >> 8). TRUNG BÌNH cho ≥30 task sequences. Kết hợp Direction 1+2 giữ null-space rộng → giảm thiểu.

#### Phản biện 3: Data-informed init = MỘT DẠNG data statistic?

$C_{\text{data}} = \mathbb{E}[h h^T]$ tính từ training data. $A_t$ = top eigenvectors của $Q C_{\text{data}} Q$. Tại inference, $A_t$ vẫn ở trong model (frozen). Nhưng $A_t$ MANG THÔNG TIN về phân phối data (covariance matrix eigenvectors ∝ principal components of data).

**So sánh với GPM**: GPM bases CŨNG tính từ activation covariance (cùng $C_{\text{data}}$ bản chất!). GPM đã được chấp nhận trong zero-replay setting (InfLoRA paper, GainLoRA paper). 

**Lập luận**: $A_t$ tính từ covariance CỦA TASK HIỆN TẠI (không phải task cũ). Trong CL, training data của task hiện tại LUÔN available. Zero-replay cấm dùng lại data CŨ, không cấm sử dụng data HIỆN TẠI.

**NHƯNG**: Tại inference (sau task t), $A_t$ vẫn encode thông tin về data task t. Đây có vi phạm không?
- GPM bases cũng encode: directions mà data task t biến động
- Prototype $\mu_t$ encode: trung tâm phân phối data task t ← BỊ CẤM
- $A_t$ (data-informed) encode: hướng variance cao nhất trong null-space theo data task t

$A_t$ encode **hướng** (không phải vị trí/giá trị cụ thể), tương tự GPM bases. Nếu GPM bases được phép, $A_t$ data-informed cũng được phép.

**Mức độ nghiêm trọng**: THẤP. Logic nhất quán với InfLoRA/GainLoRA đã được chấp nhận. Nhưng cần nêu rõ trong paper rằng data-informed init KHÔNG lưu data, chỉ lưu model parameters (A_t) derived from data.

#### Phản biện 4: Cải thiện THỰC SỰ bao nhiêu?

Claim: data-informed A_t > random A_t. Nhưng bao nhiêu?

- Nếu task-relevant directions chiếm phần lớn null-space (easy task, null-space rộng): random cũng OK → cải thiện NHỎ
- Nếu task-relevant directions chiếm phần nhỏ null-space (hard task, null-space hẹp): data-informed SỰ SỐNG ≠ CÁI CHẾT → cải thiện LỚN

V6 IMDB: null-space 351D, IMDB fail hoàn toàn (EM=0). → thuộc case 2 → data-informed có khả năng cải thiện LỚN.

Nhưng cần empirical verification. Lý thuyết chỉ nói "tối ưu trong null-space" nhưng nếu all relevant directions nằm NGOÀI null-space thì kể cả data-informed cũng fail.

**Mức độ nghiêm trọng**: TRUNG BÌNH. Cần experiment để verify. Lý thuyết supportive nhưng không conclusive.

#### Phản biện 5: Conflict với spectral routing?

Data-informed $A_t$ có hàng aligned với task data → $V_t$ (từ SVD($B_t A_t$)) cũng aligned → spectral routing TỐT HƠN cho task t.

NHƯNG: nếu IMDB data-informed A capture sentiment directions (partially in null-space) → $V_{\text{imdb}}$ overlap với sentiment → CŨNG overlap với $V_{\text{yelp}}$ (đã claim sentiment directions) → routing confusion TĂNG.

**Giải pháp**: Đây chính là GPM's job — null-space projection đảm bảo $A_{\text{imdb}}$ KHÔNG overlap với protected directions. Data-informed chọn TRONG null-space → overlap với yelp's PROTECTED directions = 0. Overlap chỉ có thể xảy ra với yelp's UNPROTECTED directions (nếu dùng Direction 1) → nhưng unprotected directions có σ nhỏ → ít ảnh hưởng routing.

**Mức độ nghiêm trọng**: THẤP. Null-space projection + selective protection tự nhiên prevent routing conflict.

#### Kết luận Round 3

**Đánh giá tổng thể**: Data-Informed Subspace Tracking là hướng **MẠNH NHẤT** trong 3 hướng.
- ✅ Trực tiếp cải thiện QUALITY of null-space usage (không chỉ quantity)
- ✅ Tấn công trực tiếp root cause (A_t random → A_t task-relevant)
- ✅ Zero-replay compliant (cùng logic với GPM)
- ✅ Tăng cường Routing-Protection Duality (better expert → better routing)
- ✅ Mathematical: Constrained PCA on projected Grassmannian — clean closed-form solution
- ⚠️ Cần warm-up forward pass (~30s overhead per task)
- ⚠️ Cần empirical verification

---

## VI. Tổng hợp Đánh giá

### Ma trận đánh giá

| Tiêu chí | Direction 1 (Selective) | Direction 2 (Defrag) | Direction 3 (Data-Informed) |
|-----------|:---:|:---:|:---:|
| Giải quyết null-space collapse | ★★★ | ★★☆ | ★★★★ |
| Giải quyết routing failure | ★☆ | ★ | ★★★ |
| Zero-replay compliance | ✅ | ✅ | ✅ |
| Lý thuyết toán học | ★★★ | ★★★ | ★★★★ |
| Tương thích Duality | ★★★★ | ★★★ | ★★★★★ |
| Dễ implement | ★★★★ | ★★★ | ★★★ |
| Mâu thuẫn C4? | ⚠️ Có (entropy reg) | ✅ Không | ✅ Không |
| Novelty | ★★★ | ★★☆ | ★★★★ |
| Tác động dự kiến | Trung bình | Nhỏ-Trung bình | Lớn |

### Kết luận tổng hợp

**Direction 3 (Data-Informed Subspace Tracking)** là hướng MẠNH NHẤT và nên là **core change cho V7**:
- Trực tiếp tấn công root cause: random A → data-informed A
- Có closed-form optimal solution (Constrained PCA)
- Tăng cường cả learning quality VÀ routing quality
- Clean mathematical narrative (optimal subspace on restricted Grassmannian)

**Direction 1 (Selective Protection)** nên là **supplementary change**:
- Giảm GPM growth rate → mở rộng null-space cho Direction 3 khai thác
- Nhưng CẦN giải quyết mâu thuẫn C4 (có thể: bỏ entropy reg, hoặc dùng fixed k thay vì σ-threshold)

**Direction 2 (Defragmentation)** nên **cân nhắc thêm**:
- Lợi ích nhỏ nếu Direction 1 đã giảm đủ GPM dims
- Thêm complexity (SVD tổng bases mỗi task) cho marginal gain
- Có thể implement sau nếu Direction 1+3 chưa đủ

### Khuyến nghị cho V7

**Primary**: Implement Direction 3 (Data-Informed Init) → thay random kaiming init bằng constrained PCA init.

**Secondary**: Xem xét Direction 1 (Selective Protection) nếu Direction 3 alone không đủ. Nhưng cần giải quyết C4 conflict trước.

**Defer**: Direction 2 (Defragmentation) — giữ làm backup plan.

### Thay đổi so với idea hiện tại

**Cần thay đổi IDEA không?** 

Direction 3 KHÔNG thay đổi theoretical framework (Routing-Protection Duality vẫn đúng). Nó chỉ thay đổi IMPLEMENTATION:
- Thay $A_t = \text{random projected to null-space}$
- Bằng $A_t = \text{top-r eigenvectors of projected covariance}$

Đây là **improvement trong implementation**, KHÔNG phải thay đổi idea.

→ **Kết luận**: Idea hiện tại CHẶT CHẼ. Direction 3 là cải tiến implementation. Phù hợp work_ethic: "chỉ cải thiện implement tốt hơn, nhưng không sa vào over engineer".

nhưng Direction 1 (nếu adopt) THAY ĐỔI protection mechanism → cần update lý thuyết (Theorem 1 với selective ε). Đây là thay đổi idea-level, cần cân nhắc kỹ.
