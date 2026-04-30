# Phân Tích 3 Phiên Bản SRT

Tài liệu này là bản tiếng Việt hoàn chỉnh của phần phân tích SRT, đồng thời có thêm phụ lục ngắn theo kiểu commit-by-commit để đối chiếu đúng thay đổi nào đã làm routing tăng.

Mục tiêu là trả lời chính xác ba câu hỏi:

1. SRT bản thấp trong `hmm_log.txt`, tức line `llama`, thực chất là gì.
2. SRT bản thử nghiệm `routing_class_v2`, thường cao hơn bản runtime một chút, là gì.
3. SRT bản gần như perfect ở line `run_debug` khác hai bản kia ở đâu, và vì sao nó cải thiện.

## Kết luận ngắn

Điểm quan trọng nhất là thế này:

- mức tăng lên gần perfect **không đến từ patch low-VRAM**,
- mức tăng chủ yếu đến từ commit `ad60437`, nơi router được đổi từ pooled covariance kiểu cũ sang pooled covariance **within-class**,
- còn line `llama` ở ref `048e693` thực ra đã có phần lớn alignment về prompt và extractor rồi.

Nói gọn hơn:

- `llama` thấp hơn không phải vì prompt/runtime quá lệch nữa,
- `run_debug` cao hơn chủ yếu vì **router math đã đúng hơn**.

## Bản đồ 3 phiên bản

| Phiên bản | Mốc cụ thể | Nó là gì | Điểm khác biệt chính |
| --- | --- | --- | --- |
| SRT bản thấp | branch `llama`, ref `048e693`, log ở `new_llama_gainlora/hmm_log.txt` | SRT runtime đã được deploy trong pipeline continual LLaMA | đã align source-only routing, nhưng covariance vẫn là kiểu pooled cũ |
| SRT bản thử nghiệm `routing_class_v2` | `routing_analysis/routing_class_v2.py`, artifact cũ `routing_analysis/ablation_truely.txt` | proxy offline trên frozen embeddings | bề mặt đánh giá sạch hơn runtime, nhưng artifact cũ vẫn dùng router math cũ |
| SRT gần perfect | branch `new_debug` / `run_debug`, head `9f023a2`, patch thuật toán nằm ở `ad60437` | runtime sau khi sửa router math | khác biệt quyết định là pooled within-class covariance và metadata router rõ ràng |

## Phần I. SRT bản thấp từ `hmm_log.txt` là gì

### 1. Nhận diện chính xác

Đây là phiên bản runtime chạy trong line `llama`.

Mốc branch hiện tại là `048e693`.

Một nhầm lẫn rất dễ xảy ra là nhìn thấy dòng `70%` trong log rồi tưởng đó là routing accuracy. Không phải.

Trong `hmm_log.txt`, các dòng kiểu:

```text
Eval @ step 25:  70%|███████   | 7/10 [...]
```

chỉ là progress bar của quá trình eval.

Routing accuracy thật nằm ở box `SRT ROUTING ACCURACY SUMMARY`.

### 2. Bản thấp thực ra thấp tới mức nào

Nếu đọc đúng summary trong `hmm_log.txt`, ta có:

- task 1: `100.0% (20/20)`
- task 2: `98.6% (138/140)`
- task 3: `97.2% (350/360)`
- task 4: `97.8% (609/623)`

Vậy nên bản này không hề tệ. Nó đã rất mạnh, chỉ là chưa full 100 ở các round sau.

### 3. Bản thấp đã có những gì rồi

Đây là điểm rất quan trọng. Khi tới `048e693`, line `llama` đã có hầu hết phần alignment lớn:

- routing dùng `input_ids_wo_label`, tức là chỉ route trên source prompt,
- prompt route là `instruction.format(sentence)`,
- tokenizer dùng `add_special_tokens=False`,
- embedding route lấy từ frozen LLaMA extractor,
- pooling dùng hidden state cuối tại vị trí non-padding cuối cùng,
- đường tạo signature support và đường runtime inference đều đã dùng cùng source-prompt profile.

Điều đó có nghĩa là bản `llama` không còn là bản runtime “lệch hẳn offline” như giai đoạn đầu nữa.

### 4. Bản thấp còn sai ở đâu

Điểm còn yếu của nó nằm chủ yếu ở **router math**.

Router ở line này vẫn dùng pooled covariance kiểu cũ. Về mặt ý tưởng, nó tương đương với covariance của **toàn bộ union sample** của các task đã thấy, thay vì covariance **within-task** dùng chung cho classifier.

Có thể hình dung như sau:

```text
Sigma_union ≈ Sigma_within + Sigma_between
```

Trong routing theo Mahalanobis, cái ta muốn là phần nhiễu within-task quanh centroid của từng task. Nếu trộn luôn cả độ tách giữa các task vào pooled covariance, ta đã làm covariance “phình” ra ở đúng những hướng đang giúp phân biệt task.

Hệ quả là Mahalanobis score bị bớt sắc nét.

### 5. Ý nghĩa thực nghiệm

Vì line `llama` đã align tốt ở prompt/extractor rồi, nên lý do nó còn thua một chút không nằm ở phần input nữa, mà nằm ở việc shared covariance vẫn đang mô hình hóa sai đối tượng thống kê.

Đó là lý do mức chênh không lớn, nhưng vẫn đủ để còn sót vài mẫu sai ở round 2, 3, 4.

## Phần II. SRT bản thử nghiệm `routing_class_v2` là gì

### 1. Bản chất của `routing_class_v2`

`routing_analysis/routing_class_v2.py` là một bộ đánh giá routing **offline** trên frozen embeddings `.npz`.

Nó không phải runtime inference thực sự đang chạy bên trong `model.generate(...)`.

Nó là công cụ kiểm định giả thuyết và đóng vai trò proxy cho runtime.

### 2. Vì sao nó thường cao hơn runtime một chút

Phiên bản này thường cho cảm giác “đẹp” hơn runtime vì nó loại bỏ nhiều tầng nhiễu của hệ runtime:

- không phải đi qua generation loop,
- không bị xen bởi plumbing của continual prediction,
- không dính hành vi của `generate(...)`,
- làm việc trực tiếp trên frozen embeddings đã trích sẵn.

Nói ngắn gọn, đây là bề mặt đánh giá sạch hơn.

Ngoài ra, nhiều kết quả offline được trình bày dưới dạng macro theo task, trong khi summary runtime thường là micro accuracy trên pool seen-task của vòng đó. Hai kiểu đo này không hoàn toàn giống nhau.

### 3. Vì sao nó chỉ “cao hơn một chút” chứ chưa perfect hẳn

Artifact hiện có trong repo là `routing_analysis/ablation_truely.txt`.

File này là **artifact cũ**, được tạo ra trước khi router math được sửa sang within-class.

Do đó, trong bối cảnh lịch sử của artifact này, `routing_class_v2` vẫn dùng chung điểm yếu toán học với bản runtime thấp:

- pooled covariance kiểu cũ,
- shrinkage theo `n_pool`,
- chưa có `cov_dof`,
- chưa có `covariance_mode`.

Tức là nó sạch hơn về mặt bề mặt đánh giá, nhưng chưa sạch hơn về mặt estimator.

### 4. Tại sao không được lấy `ablation_truely.txt` làm chân lý hiện tại

Hiện nay source `routing_analysis/routing_class_v2.py` đã được patch theo within-class covariance.

Nhưng `ablation_truely.txt` chưa được regenerate.

Vì vậy cặp này hiện đang lệch nhau:

- source hiện tại: math mới,
- artifact text trong repo: math cũ.

Nên nếu đem số runtime mới so thẳng với `ablation_truely.txt` thì rất dễ kết luận sai.

## Phần III. SRT gần perfect ở `run_debug` là gì

### 1. Tên branch dễ gây hiểu nhầm

Nếu nhìn theo tên branch, ta dễ tưởng line `run_debug` mạnh lên vì các thay đổi ở branch này.

Nhưng thực ra phải tách ra làm hai lớp:

- patch thuật toán quan trọng: `ad60437`,
- patch low-VRAM phía trên: `692ed0a`,
- head hiện tại của branch: `9f023a2`.

Trong các file lõi của routing, phần làm accuracy tăng nằm ở `ad60437`, không nằm ở low-VRAM wrapper.

### 2. Khác biệt cốt lõi so với `llama`

Từ `048e693` sang `ad60437`, diff liên quan routing chỉ tập trung vào ba file:

- `new_llama_gainlora/src/srt_router_v2.py`
- `new_gainlora/src/srt_router.py`
- `routing_analysis/routing_class_v2.py`

Điều này cực kỳ quan trọng: nó cho thấy bước nhảy accuracy đến từ **router estimator**, chứ không phải từ một thay đổi lặt vặt ở logging hay launcher.

### 3. Router math mới là gì

Router mới đưa vào các khái niệm sau:

- `COVARIANCE_MODE = "within_class"`
- `cov_dof`
- `welford_pooled_update(..., dof_old, ..., n_new)`
- shrinkage dùng `max(cov_dof, 1)` thay vì `n_pool`
- state save/load có thêm `cov_dof` và `covariance_mode`
- từ chối load router artifact kiểu legacy cũ.

Covariance mới có nghĩa là:

```text
Sigma_within = sum_t ((n_t - 1) * Sigma_t) / sum_t (n_t - 1)
```

Trong khi pooled mean vẫn là:

```text
mu_pool = sum_t (n_t * mu_t) / sum_t n_t
```

Đây mới là object thống kê đúng cho pooled-covariance Mahalanobis classifier khi mỗi task có centroid riêng nhưng chia sẻ covariance nền.

### 4. Vì sao nó có thể nhảy từ 97-98 lên 100

Khi dùng covariance kiểu union cũ, khoảng cách giữa các task bị “ăn mòn” một phần vì covariance đã chứa luôn giữa-task scatter.

Khi chuyển sang within-class covariance:

- các hướng phân tách giữa task không còn bị xem là nhiễu chung,
- inverse covariance sắc hơn,
- Mahalanobis distance phân tách centroid tốt hơn,
- vài lỗi biên cuối cùng có thể biến mất.

Trong tình huống round đầu hoặc round trung bình, nơi các task vốn đã khá tách nhau, việc xóa nốt vài lỗi là hoàn toàn có thể dẫn tới 100%.

### 5. Điều gì không phải nguyên nhân

Không phải các thay đổi low-VRAM.

Patch `692ed0a` chỉ đụng:

- default eval batch size,
- gradient checkpointing,
- DeepSpeed config của launcher/runtime.

Nó không đụng vào các file router lõi.

Do đó, nếu hỏi “điều gì làm routing gần perfect”, thì câu trả lời đúng là **patch within-class covariance**, chứ không phải patch low-VRAM.

## Phần IV. So sánh ngắn gọn ba bản

| Trục so sánh | Bản thấp `llama` | Bản thử nghiệm `routing_class_v2` | Bản gần perfect `run_debug` |
| --- | --- | --- | --- |
| Môi trường đánh giá | runtime thật | offline trên frozen embeddings | runtime thật |
| Đầu vào route | source-only prompt | embedding trích sẵn | source-only prompt |
| Độ sạch của bề mặt đánh giá | thấp hơn offline | sạch hơn runtime | thấp hơn offline nhưng thực chiến hơn |
| Covariance dùng cho Mahalanobis | pooled cũ kiểu union | artifact cũ cũng là union | pooled within-class |
| Shrinkage calibration | theo `n_pool` | theo `n_pool` ở artifact cũ | theo `cov_dof` |
| Ý nghĩa của số đo | số deploy runtime | proxy kiểm định giả thuyết | số deploy runtime |

## Phần V. Kết luận thực chất

Nếu bỏ qua tên branch và chỉ nhìn vào logic thống kê, thì câu chuyện là:

1. bản `llama` thấp hơn một chút vì pooled covariance vẫn đang trộn cả between-task scatter,
2. `routing_class_v2` là một bề mặt offline sạch hơn nên thường nhỉnh hơn một chút, nhưng artifact cũ vẫn chưa phản ánh math mới,
3. bản gần perfect mạnh lên chủ yếu vì pooled covariance đã được sửa thành within-class covariance.

Đó là nguyên nhân hợp lý nhất, trực tiếp nhất, và được hỗ trợ rõ nhất bởi diff code.

---

## Phụ lục A. Timeline commit-by-commit

Phụ lục này chỉ giữ các commit thực sự liên quan đến câu chuyện SRT, không cố liệt kê toàn bộ lịch sử repo.

### Commit `8ca0a47` — bước refactor runtime SRT lớn

Thời gian: `2026-04-29 00:11:59 +0700`

Các file chính bị chạm:

- `new_llama_gainlora/run_llama_gainlora_cl.py`
- `new_llama_gainlora/src/llama_gainlora.py`
- `new_llama_gainlora/src/sgwi_srt_trainer.py`

Ý nghĩa:

- thêm đường runtime route trên source-only input,
- đưa `input_ids_wo_label` vào pipeline,
- thêm model runtime mới trong `llama_gainlora.py`,
- củng cố support-signature extraction.

Tác động tới routing:

- rất quan trọng về mặt plumbing và alignment,
- nhưng đây chưa phải patch within-class covariance.

Đánh giá vai trò:

- đây là commit dựng nền cho SRT runtime đúng profile,
- chưa phải commit giải thích bước nhảy cuối cùng lên near-perfect.

### Commit `f537265` — sửa bug load module

Thời gian: `2026-04-29 01:26:11 +0700`

File chính bị chạm:

- `new_llama_gainlora/run_llama_gainlora_cl.py`

Ý nghĩa:

- thêm `sys.modules[module_name] = module` trong loader.

Tác động tới routing:

- gần như không phải thay đổi thuật toán routing,
- đây là fix ổn định hóa import/module loading.

Đánh giá vai trò:

- không phải nguyên nhân làm accuracy tăng.

### Commit `048e693` — mốc tip của line `llama`

Thời gian: `2026-04-29 11:09:35 +0700`

File chính bị chạm:

- `new_llama_gainlora/run_llama_gainlora_cl.py`

Ý nghĩa cụ thể:

- commit này chủ yếu comment-out các dòng debug log `SRT-D` rất ồn,
- không đổi router math,
- không đổi covariance estimator.

Tác động tới routing:

- gần như không đổi accuracy thực,
- chủ yếu đổi mức ồn của log.

Đánh giá vai trò:

- đây là mốc branch để đại diện cho “bản thấp” mà bạn đang so,
- nhưng không phải commit tạo ra bản thấp đó.

### Commit `ad60437` — patch quyết định

Thời gian: `2026-04-29 18:13:37 +0700`

Các file chính bị chạm:

- `new_llama_gainlora/src/srt_router_v2.py`
- `new_gainlora/src/srt_router.py`
- `routing_analysis/routing_class_v2.py`

Ý nghĩa cụ thể:

- thêm `COVARIANCE_MODE = "within_class"`,
- thêm `cov_dof`,
- đổi `welford_pooled_update` sang dạng có `dof_old`,
- shrinkage dùng `cov_dof` thay vì `n_pool`,
- save/load state có thêm metadata để chặn nhầm artifact legacy.

Tác động tới routing:

- đây là thay đổi thuật toán mạnh nhất,
- đây là ứng viên số 1 giải thích việc routing từ khoảng 97-98% nhảy lên rất gần hoặc chạm 100%.

Đánh giá vai trò:

- **đây là commit quan trọng nhất của toàn bộ câu chuyện cải thiện**.

### Commit `692ed0a` — patch low-VRAM

Thời gian: `2026-04-29 19:07:49 +0700`

Các file chính bị chạm:

- `new_llama_gainlora/run_llama_gainlora_cl.py`
- `new_llama_gainlora/run_superni_order1_llama_cl.sh`

Ý nghĩa cụ thể:

- đổi default eval batch,
- đổi default gradient checkpointing,
- đổi default DeepSpeed config sang CPU offload.

Tác động tới routing:

- không chạm vào router estimator,
- không chạm vào `srt_router_v2.py`,
- không chạm vào `routing_class_v2.py`.

Đánh giá vai trò:

- không phải nguyên nhân làm routing gần perfect,
- chỉ là patch bộ nhớ cho thử nghiệm.

### Commit `9f023a2` — head của `new_debug`

Thời gian: `2026-04-29 19:25:37 +0700`

Ý nghĩa:

- đây là head hiện tại của line `new_debug`,
- trong các file routing lõi đang xét, không có thay đổi thuật toán mới nào sau `692ed0a`.

Đánh giá vai trò:

- nó là mốc branch/head để bạn quan sát run gần perfect,
- nhưng patch làm accuracy tăng vẫn là `ad60437`.

---

## Phụ lục B. Câu trả lời ngắn nhất có thể

Nếu phải tóm lại trong ba câu:

1. Bản `llama` đã đúng hơn rất nhiều so với giai đoạn đầu, nhưng covariance còn sai kiểu thống kê.
2. Bản `routing_class_v2` là proxy offline sạch hơn, nhưng artifact cũ trong repo vẫn là math cũ.
3. Bản gần perfect mạnh lên chủ yếu vì `ad60437` đổi pooled covariance sang within-class covariance; patch low-VRAM không phải nguyên nhân chính.

---

## Phụ lục C. Bảng So Sánh Kỹ Thuật Chi Tiết

| Trục kỹ thuật | Bản thấp `llama` / `hmm_log` | Bản thử nghiệm `routing_class_v2` | Bản gần perfect `run_debug` | Ảnh hưởng tới cải thiện |
| --- | --- | --- | --- | --- |
| Bề mặt đánh giá | Runtime thật, route trong vòng continual prediction | Offline trên frozen embeddings `.npz` | Runtime thật, route trong vòng continual prediction | Offline thường sạch hơn runtime, nhưng không phải nguyên nhân chính của bước nhảy cuối |
| Tập mẫu được chấm | Micro accuracy trên pool seen-task của vòng hiện tại | Thường báo macro theo task và tách theo round | Micro accuracy trên pool seen-task của vòng hiện tại | Khác metric surface có thể làm số trông hơi lệch, nhưng không giải thích 97-98 lên 100 |
| Query dùng để route | Source-only prompt qua `input_ids_wo_label` | Embedding frozen đã được trích sẵn | Source-only prompt qua `input_ids_wo_label` | Giữa `llama` và `run_debug` phần này gần như không còn là khác biệt lớn |
| Prompt route | `instruction.format(sentence)` | Phụ thuộc profile lúc trích embedding; artifact cũ là profile lịch sử | `instruction.format(sentence)` | Khác biệt này quan trọng khi so runtime với offline cũ, nhưng không còn là khác biệt chính giữa `llama` và `run_debug` |
| Tokenization route | `add_special_tokens=False` | Phụ thuộc metadata embedding; artifact cũ không nên xem là chuẩn hiện tại | `add_special_tokens=False` | Chủ yếu giải thích vì sao runtime mới đã gần offline hơn từ trước khi đổi math |
| Pooling embedding | Hidden state cuối tại last non-padding token | Dùng embedding đã frozen sẵn | Hidden state cuối tại last non-padding token | Không phải tác nhân chính của bước nhảy cuối nếu so `llama` với `run_debug` |
| Nguồn embedding support | Frozen backbone, trích từ train samples của task hiện tại | Train embeddings `.npz` đã lưu | Frozen backbone, trích từ train samples của task hiện tại | Hai runtime gần như cùng profile, nên đây không phải chỗ tạo ra near-perfect |
| Covariance dùng cho Mahalanobis | Pooled covariance kiểu cũ, gần với union covariance | Artifact cũ cũng là pooled covariance kiểu cũ | Pooled within-class covariance | Đây là khác biệt quan trọng nhất và là ứng viên số 1 giải thích việc điểm tăng |
| Công thức pooled covariance | Trộn cả within-task và một phần between-task scatter | Artifact cũ cũng như vậy | Chỉ gom within-task covariance với `sum (n_t - 1)` | Khi bỏ được between-task scatter khỏi shared covariance, biên phân tách task sắc hơn rõ rệt |
| Tham số shrinkage | Dựa trên `n_pool` | Dựa trên `n_pool` trong artifact cũ | Dựa trên `cov_dof` | Làm inverse covariance được hiệu chỉnh đúng hơn, nhất là khi số task tăng |
| Metadata router state | Cũ hơn, không tự mô tả rõ semantics covariance | Không phải runtime state | Có `cov_dof` và `covariance_mode` | Giảm nguy cơ load nhầm state legacy rồi tưởng đang đánh giá math mới |
| Tương thích artifact cũ | Có thể vô tình sống chung với semantics cũ | Artifact cũ chính là semantics cũ | Từ chối `union_legacy` khi load | Tăng độ tin cậy khoa học, vì runtime mới không lẫn state cũ |
| Vai trò của commit `8ca0a47` | Đã hấp thụ phần lớn plumbing alignment trước khi tới `048e693` | Không trực tiếp là artifact offline cũ | Là nền có sẵn trước patch math | Rất quan trọng để route đúng profile, nhưng không phải cú nhảy cuối lên near-perfect |
| Vai trò của commit `ad60437` | Chưa có | Chưa được reflect trong `ablation_truely.txt` | Đã có | Đây là commit làm estimator đổi thật sự |
| Vai trò của patch low-VRAM `692ed0a` | Chưa có | Không liên quan | Có trong branch `run_debug` | Không phải nguyên nhân chính của việc routing tăng |
| Tình trạng artifact trong repo | `hmm_log.txt` phản ánh runtime cũ hơn | `ablation_truely.txt` là artifact cũ, chưa regenerate | Chưa có artifact offline tương ứng trong repo local | Đây là lý do không nên so thẳng runtime mới với `ablation_truely.txt` |
| Kết luận theo từng bản | Runtime đã khá chuẩn nhưng estimator còn hơi blur | Proxy offline sạch hơn nhưng artifact cũ vẫn dùng math cũ | Runtime + estimator đúng hơn nên có thể chạm near-perfect | Bước nhảy lớn nhất đến từ sửa covariance, không phải từ launcher hay logging |

---

## Phụ lục D. Phân Loại Theo 4 Khả Năng Tất Định

Phần này viết lại toàn bộ tranh luận theo đúng khung bạn nêu:

1. hoặc là input không đồng nhất tuyệt đối,
2. hoặc là implement của SRT không trùng khớp,
3. hoặc là suy luận về tính tất định là sai,
4. hoặc là API function hay cách tính tensor, vector, ma trận khác nhau.

Tôi giữ kết luận rất chặt: với case đang xét giữa `routing_class_v2.py` hiện tại và runtime trên line `llama`, **không cần viện tới "nhiễu" hay "không tất định" để giải thích**. Chỉ cần case 1, case 2 và case 4 là đã đủ.

### Case 1. Input không đồng nhất tuyệt đối

Kết luận: **chưa bị falsify, và là khả năng có thật**.

Lý do:

- runtime line `llama` route trên source-only prompt được tạo trực tiếp từ `instruction.format(sentence)`,
- tokenizer runtime dùng `add_special_tokens=False`,
- query runtime được route qua `input_ids_wo_label`, tức đúng source prompt,
- nhưng `routing_class_v2.py` không route trên raw sample trực tiếp; nó đọc embedding từ `.npz` đã trích sẵn.

Điểm then chốt là `routing_class_v2.py` hiện có logic sau:

- nếu `.npz` có `metadata_json`, nó đọc profile thật,
- nếu `.npz` **không có metadata**, nó dùng `infer_legacy_embedding_profile(...)` để **suy đoán** profile runtime mặc định.

Điều này có nghĩa là:

- nếu bạn đang dùng `.npz` cũ không có metadata,
- thì offline evaluator **không thể chứng minh** rằng input embedding của nó được trích từ đúng cùng prompt/profile với runtime `llama`.

Nói cách khác, case 1 hoàn toàn chưa bị loại trừ.

Điểm rất quan trọng ở đây là: việc source code hiện tại "infer" ra profile `runtime_cl`, `add_special_tokens=False`, `max_length=1024` không làm cho `.npz` cũ tự động trở thành embedding mới. Nó chỉ làm checker bớt chặn chạy.

### Case 2. Implement của SRT không trùng khớp

Kết luận: **đã được xác nhận chắc chắn**.

Đây là điểm cứng nhất trong toàn bộ phân tích.

So trực tiếp giữa runtime router trên line `llama` và `routing_class_v2.py` hiện tại:

- runtime `llama` cũ dùng `welford_pooled_update(mu_old, cov_old, n_old, mu_new, cov_new, n_new)`,
- `routing_class_v2.py` hiện tại dùng `welford_pooled_update(mu_old_t, cov_old_t, n_old, dof_old, mu_new_t, cov_new_t, n_new)`.

Khác biệt không phải tên hàm, mà là nội dung toán bên trong.

Runtime `llama` cũ dùng cập nhật pooled covariance kiểu union, có thêm cross-term giữa các mean:

```text
cross = (n_old * n_new / total) * outer(delta, delta)
cov_pool = ((n_old - 1) * cov_old + (n_new - 1) * cov_new + cross) / (total - 1)
```

Trong khi `routing_class_v2.py` hiện tại dùng pooled within-class covariance:

```text
total_dof = dof_old + dof_new
cov_pool = ((dof_old * cov_old_t) + (dof_new * cov_new_t)) / total_dof
```

Tức là hai router này **không phải cùng một estimator** nữa.

Vì vậy nếu một bên 97-98% còn một bên 100%, đó không phải điều lạ. Đó đơn giản là vì bạn đang so hai implementation khác nhau.

### Case 3. Suy luận về tính tất định là sai

Kết luận: **về mặt thực chất, không**.

Tôi đồng ý với lập luận của bạn ở mức cốt lõi:

- nếu input giống hệt,
- nếu implementation giống hệt,
- nếu cùng công thức tensor/matrix,
- nếu cùng artifact router state,

thì SRT phải cho cùng output theo tính tất định của hàm.

Trong tranh luận này, không cần dựa vào một giả thuyết kiểu "có nhiễu nên route dao động" để giải thích.

Nuance duy nhất là: trên GPU, một số phép toán floating-point có thể không bitwise-identical tuyệt đối do thứ tự reduction/song song. Nhưng những sai lệch kiểu đó không phải cách hợp lý để giải thích một thay đổi có cấu trúc như:

- cũ: còn sót vài lỗi 97-98%,
- mới: full 100,
- đồng thời code estimator đã đổi thật.

Nói ngắn gọn:

- giả thuyết "SRT không tất định" không cần dùng,
- và cũng không phải giả thuyết tốt nhất để giải thích hiện tượng đang thấy.

### Case 4. API function hay cách tính tensor, vector, ma trận khác nhau

Kết luận: **đúng, và đây chính là dạng cụ thể của case 2**.

Nếu muốn nói ở mức matrix calculus thay vì mức "implementation", thì khác biệt nằm đúng ở đây:

#### 4.1. Công thức pooled covariance khác nhau

Runtime `llama` cũ:

- dùng cross-term giữa mean cũ và mean mới,
- ra covariance kiểu union.

`routing_class_v2.py` hiện tại:

- bỏ hoàn toàn cross-term đó,
- chỉ trung bình các covariance within-class theo bậc tự do.

Đây là khác biệt ma trận cấp 1, không phải tiểu tiết lập trình.

#### 4.2. Đối số shrinkage khác nhau

Runtime `llama` cũ:

```text
Sigma_shrunk = shrink_fn(Sigma_pool_t, n_pool, d)
```

`routing_class_v2.py` hiện tại:

```text
Sigma_shrunk = shrink_fn(Sigma_pool_t, max(cov_dof, 1), d)
```

Tức là ngay cả khi cùng gọi `ridge`, `oas`, `lw`, tham số đi vào shrinkage cũng đã khác.

#### 4.3. Router state semantics khác nhau

Router hiện tại có thêm:

- `cov_dof`,
- `covariance_mode`,
- logic chặn load `union_legacy`.

Điều này đảm bảo runtime mới và offline mới không vô tình dùng chung một artifact cũ rồi tưởng là cùng semantics.

### Bảng chốt 4 khả năng

| Khả năng | Trạng thái | Kết luận ngắn |
| --- | --- | --- |
| 1. Input không đồng nhất tuyệt đối | Chưa bị loại trừ | Có khả năng đúng, nhất là nếu offline đang dùng `.npz` legacy hoặc artifact cũ |
| 2. Implement SRT không trùng khớp | Đã xác nhận | Đúng chắc chắn giữa `routing_class_v2.py` hiện tại và router trên line `llama` |
| 3. SRT không tất định | Gần như bị loại | Không cần giả thuyết này để giải thích hiện tượng đang thấy |
| 4. API / matrix computation khác nhau | Đã xác nhận | Đúng, và chính là biểu hiện cụ thể của case 2 |

### Kết luận cuối cùng theo khung 4 khả năng

Nếu buộc phải kết luận thật gọn và thật cứng, thì câu trả lời là:

- **case 3 không phải lời giải**,
- **case 2 và case 4 chắc chắn đúng**,
- **case 1 nhiều khả năng cũng chưa được loại trừ**, đặc biệt khi so với artifact offline cũ.

Vì vậy, lời giải bảo thủ và kỹ thuật đúng nhất là:

> SRT vẫn là hàm tất định. Mismatch không đến từ "nhiễu". Mismatch đến từ việc input/offline artifact chưa được chứng minh là đồng nhất tuyệt đối, và quan trọng hơn là implementation/matrix update của router hiện tại trong `routing_class_v2.py` không còn trùng với router trên line `llama` nữa.