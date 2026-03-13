# Giải mã sự khác biệt về VRAM và Thời gian huấn luyện (GainLoRA vs SpecRoute)

Chào bạn, dựa trên câu hỏi của bạn, tôi đã phân tích 2 file kịch bản huấn luyện:
`gen_script_long_order3_t5_gainlora_inflora.sh` (chứa thuật toán **GainLoRA gốc**)
và `gen_script_long_order3_t5_specroute.sh` (chứa thuật toán **SpecRoute cải tiến**).

Đầu tiên để trả lời câu hỏi của bạn: **Đúng, chúng hoàn toàn CÙNG HỆ QUY CHIẾU.**
- **Cùng dataset:** 15 tasks (`yelp`, `amazon`, `mnli`, `cb`, `copa`, ...)
- **Cùng Backbone:** T5 model (`--model_name_or_path`)
- **Cùng Hyperparameters cốt lõi:** LR = 0.0003, LoRA r = 8, Alpha = 32. Epochs = 10, max_seq_len = 512, target_len = 50.

Vậy tại sao **VRAM usage** và **Thời gian chạy từng task (Time/Task)** lại khác nhau một trời một vực? Cốt lõi nằm ở **3 thay đổi mang tính hệ thống** trong cơ chế hoạt động của thuật toán và các setup kỹ thuật môi trường. Mời bạn xem phân tích kỹ thuật dưới đây:

## 1. Sự khác biệt về chiến lược phân bổ Batch Size và GPU (Nguyên nhân trực tiếp đổi VRAM/Thời gian chờ)
Nhìn vào 2 file bash, ta thấy cách cấp phát tham số lô dữ liệu (Batch Size) cho từng task bị thay đổi:

*   **GainLoRA gốc (T5_GainLoRA_InfLoRA):**
    *   Sử dụng thông số cứng (hardcoded): `per_device_train_batch_size 16`, `gradient_accumulation_steps 2`.
    *   Nghĩa là nó gộp $16 \times 2 = 32$ mẫu/bước cập nhật. Điều này đòi hỏi VRAM cố định rất lớn. Bắt buộc user phải chạy trên card khủng cỡ con hàng **A100-80GB** (như comment out trên đầu file).
*   **SpecRoute cải tiến:**
    *   Tác giả đã cài cắm một đoạn script Bash auto-detect (phát hiện tự động) GPU ở dòng 14-55 để tự co giãn cấu hình tùy vào dung lượng VRAM:
        *   Nếu nó nhận diện ra bạn chỉ có **card T4 (~16GB)**: Nó sẽ bóp nhỏ batch size xuống `BSZ=4`, đồng thời trải rộng `GA=8` (tổng vẫn là 32) và nó đặc biệt cưỡng ép bật tính năng bật `--gradient_checkpointing` để bảo toàn RAM bộ nhớ.
        *   Việc giảm BatchSize xuống làm giảm VRAM tiêu thụ nhưng bù lại GPU sẽ rảnh rỗi hơn để tính Gradient (chậm hơn đi nhiều).
        
**👉 Kết luận 1:** SpecRoute được thiết kế lại để bóp VRAM cho vừa với túi tiền sinh viên/server nghèo (T4 GPUs), dẫn đến đánh đổi việc huấn luyện lâu hơn một chút (do batch_size bị thu nhỏ).

## 2. Gánh nặng Memory Replay vs Routing Phi tham số (Bản chất thuật toán)
Sự khác biệt lớn nhất tạo nên Performance leap (bước nhảy vọt hiệu năng) nằm ở chính lõi Python của mô hình `src/t5_specroute.py` so với file GainLoRA cũ:

*   **GainLoRA gốc (`--add_instruction_replay` & `--kl_ratio 0.1`):** 
    Phiên bản gốc dùng một cổng tham số (Parametric Learned Gating) tên là `Trans_input` (một mạng MLP nhỏ) để quyết định chia route cho các adapter. Quá trình học Cổng này rất dễ bị quên (Forgetting) nên GainLoRA **BẮT BUỘC phải dùng Data Replay/Memory Replay** + tính thêm một hàm mất mát `KL Divergence Loss` so sánh với teacher từ task cũ.
    *   **Hậu quả:** Kéo theo khối lượng tính toán khổng lồ vì phải gọi lại dữ liệu quá khứ, chạy Forward/Backward đi qua nhiều sub-networks.
*   **SpecRoute (`--run_single True` - không replay):** 
    Bản SpecifiesRoute đã thông minh **cắt bỏ hoàn toàn** mạng nơ-ron dẫn đường (no gating network MLP). Thay vào đó, nó định tuyến tĩnh bằng cách chiếu Tensor Input (phép $X \cdot V^T$) lên **Ma trận chữ ký phân bố SVD (Spectral Signatures)** của chính trọng số cũ. Nghĩa là: KHÔNG CÓ THAM SỐ NÀO CẦN HỌC cho việc Routing! KhÔNG HỌC THÌ SẼ KHÔNG QUÊN!
    *   **Hậu quả:** Nó đã loại bỏ được cờ `--add_instruction_replay`. Không replay dữ liệu cũ → Tốc độ hoàn thành 1 task nhánh tăng vọt bất chấp batch size nhỏ. VRAM cũng không phải chứa thêm đồ thị đạo hàm (computational graph) cho KL Loss.

## 3. Bottleneck ở SVD Validation (Giải thuật)
Như đã nhắc ở Báo cáo số 1, quá trình `get_representation` ở phiên bản Cũ để chống quên (Orthogonal Constraint) cần xử lý các ma trận rất dễ khựng.
*   Ở bản cũ, SVD lỗi → đẩy GPU tensor về CPU để giải mã SVD → đầy đường truyền PCI-e bus → chậm.
*   Ở `SpecRoute`, anh đã vector hóa (Batched Matrix Multiplication) kết hợp Tikhonov Regulate (`p_inv(A) + 1e-4*I`). Mọi thứ được giải quyết 100% in-situ trên chip GPU, nên mỗi Step trôi qua nhanh như một cái chớp mắt so với CPU push-pull.

---
**TÓM LƯỢC CHO BẠN DỄ HIỂU NHẤT:**
Mặc dù bạn thấy 2 file `.sh` nhận chung 1 task và 1 data, nhưng `SpecRoute` đã được thiết kế lại:
1. **Ép xung xuống cho máy yếu**: Tự động giảm `Batch_Size` và bật `Gradient Checkpointing` nếu nhận diện máy bạn xài card T4 (tiết kiệm VRAM triệt để).
2. **Loại bỏ cái cưa Data Replay**: Nhờ đổi từ *Gating ẩn* sang *Hình học phổ SVD*, SpecRoute **vứt bỏ** hẳn khâu Load data cũ và tính KL Loss tốn VRAM/vòng FORWARD.
3. **GPU SVD nguyên khối**: Không còn trò luân chuyển ma trận Tensor qua lại giữa GPU và RAM máy tính mỗi lần sinh SVD suy biến nữa.

Đó chính là câu trả lời triệt để vì sao có con số VRAM và Epoch Time chênh lớn đến thế!
