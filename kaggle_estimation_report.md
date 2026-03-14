# Ước tính phần cứng và Thời gian chạy cho `improve_gainlora`
*(Chạy Script: `gen_script_long_order3_t5_specroute.sh` trên môi trường Kaggle P100/T4 - 16GB VRAM)*

---

### 1. Phân bổ dữ liệu (Data Workload)
Sau khi quét thư mục `CL_Benchmark/Long_Sequence`, tổng khối lượng dữ liệu cho chuỗi 15 tasks (Yelp đến WiC) như sau:
*   Tổng số bản ghi (Samples): **222,718 samples**
*   Cấu hình Epochs: **10**
*   Tổng khối lượng cần xử lý: `222,718 × 10 = ~2.22 triệu samples`

### 2. Ước lượng VRAM tiêu thụ (Peak VRAM)
Máy Kaggle GPU (T4 hoặc P100 đơn) được cấp **16GB VRAM**.
*   Khi chạy kịch bản `SpecRoute`, đoạn mã auto-detect sẽ phát hiện phần cứng và thiết lập: `Batch_Size (BSZ) = 4` kèm theo `--gradient_checkpointing`.
*   Mạng T5-Base (220M params) được tải lên GPU ở dạng FP32 (Full Precision). Vì anh chỉ train trọng số **LoRA (Rank 8)** nên quá trình tính Gradient cho Backbone T5 bị đóng băng (Frozen).
*   **Chi tiết Peak VRAM** ước tính cho 1 Step (Forward + Backward) với BS = 4, chuỗi = 512:
    *   Trọng số lõi FP32 T5-Base: `~880 MB`
    *   Trạng thái Optimizer (AdamW momentum & variance) cho LoRA weights: `< 20 MB`
    *   Kích thước Tensor Activations (Kích thước Batch 4 × Độ dài 512, có bật nén Gradient Checkpointing): `~3 GB - 4.5 GB`
    *   CUDA Context / Overhead Pytorch mặc định: `~1.5 GB - 2 GB`
*   **Dự đoán Peak VRAM Thực tế (Đã cộng dư rủi ro):** Sẽ dao động ở mức **7.5 GB - 9 GB**.  Anh vẫn còn trống tới gần một nửa bộ nhớ (7GB dư dả) trên card 16GB. Tuyệt đối không bao giờ chạm ngưỡng 16GB để sinh ra lỗi OOM.

---

### *📌 Phụ lục: Tăng Batch Size có làm GIẢM thời gian huấn luyện không?*
**Câu trả lời ngắn: CÓ, nhưng hiệu quả phụ thuộc vào phần cứng (Đặc biệt với P100/T4 là RẤT ÍT).**

**Giải thích chi tiết:**
*   **Tại sao Tăng BS lại nhanh hơn?** Khi tăng Batch Size, GPU được nhồi nhiều dữ liệu hơn để thực hiện phép nhân ma trận (Matrix Multiplication) cỡ lớn song song trong 1 lần, tận dụng tối đa số lượng hàng ngàn CUDA Cores đang nằm rảnh rỗi. CPU cũng ít phải vất vả ra lệnh (kernel launching) hơn.
*   **Nhưng trên Kaggle (P100/T4) thì sao?** 
    *   **Giới hạn VRAM:** Mức hiện tại là BS=4 (đã tốn ~8.5GB). Nếu anh tăng lên BS=8, Peak VRAM sẽ nhảy lên mức **13GB - 15GB** (rất sát vách 16GB, dễ sập OOM nếu PyTorch gom rác không kịp).
    *   **Băng thông (Memory Bandwidth):** P100 là card thế hệ cũ (Pascal kiến trúc 2016), tốc độ bộ nhớ HBM2 và sức mạnh tính toán FP32 (9.3 TFLOPS) khá thấp so với A100. Việc tăng BS lên 8 thay vì 4 có thể giúp giảm được khoảng **10% - 15% tổng thời gian** (từ 35h xuống còn ~30h), nhưng sẽ **tăng rủi ro OOM lên gấp 3 lần**.
    *   Tăng Batch Size trên A100 (từ 16 lên 64) thì giảm thời gian cực kì rõ rệt (có thể giảm 1 nửa thời gian), nhưng trên P100/T4 thì CPU -> GPU bottleneck mới là thứ làm chậm.
*   **Kết luận:** Anh **không nên** mạo hiểm tăng Batch Size (BS) lên quá 4 trên môi trường Kaggle 16GB này chỉ để đổi lấy vài tiếng nhanh hơn, rủi ro văng OOM ở task số 10 rồi phải train lại từ đầu là rất cay đắng. Cấu hình tự động `BS=4, GA=8` của SpecRoute hiện tại là "ĐIỂM NGỌT" (Sweet Spot) nhất cho sự Ổn định!

### 3. Ước lượng Thời gian chạy (Execution Time)
*   Card **P100 (9.3 TFLOPS)** không có lõi Tensor Core cho FP16, nên tốc độ xử lý FP32 thực tế dao động khoảng **`15 - 20 samples / giây`** (Đã tính độ trễ khi chạy nén Gradient Checkpointing).
*   **Tính toán Thời gian:**
    *   Tổng thời gian Model Forward/Backward: `2,220,000 samples / 18 samples/s = 123,333 giây (khoảng 34 giờ)`
    *   Overhead từ validation và sinh Ma trận SVD mỗi chặng (15 task): Khoảng `1 - 2 giờ`
    *   **=> Tổng thời gian ước lượng (Total ETA): 35 đến 40 Giờ** (tuỳ thuộc vào I/O disk của môi trường Kaggle).

### 4. ⚠️ LỜI KHUYÊN QUAN TRỌNG CHO KAGGLE
*   **Kaggle có giới hạn Max Session là 12 Giờ!** Trong khi kịch bản này chạy ước tính tới  >30 giờ mới xong 15 task.
*   Nếu anh gõ lệnh `bash gen_script...` và ném đó, nó sẽ bị ngắt giữa chừng ở Task thứ 5 hoặc thứ 6 (tầm 12h) và kaggle sẽ tự động xoá Session.
*   **Phương án xử lý:** 
    1. Chia nhỏ file lệnh bash `gen_script_long_order3` thành 3 file nhỏ (Mỗi file chạy 5 task).
    2. Chạy xong 1 file thì commit lưu Checkpoint ra Google Drive (hoặc thư mục /kaggle/working bền vững) rồi mới bật Session mới chạy tiếp. 
    3. *Kịch bản bash của anh đã có sẵn lệnh `--previous_lora_path` rồi nên việc chia nhỏ nối tiếp là hoàn toàn khả thi.*
