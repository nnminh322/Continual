# Paper 01: Exploiting Presentative Feature Distributions for PE-CL of LLMs
- **Venue**: ICML 2025
- **Authors**: Xin Cheng, Jiabo Ye, Haiyang Xu, Ming Yan, Ji Zhang, Feng Liu, Fei Huang, Lei Feng
- **Link**: https://openreview.net/forum?id=6udKBHc0Mr

## Tóm tắt
Paper giải quyết vấn đề "information leakage" (IL) trong CL cho LLMs - khi task-related info của tasks cũ bị truy cập lại. Phương pháp:
1. Mỗi PEFT (LoRA) block được đặc trưng bởi "presentative feature distribution" - trung bình features từ pre-trained LLM
2. Khi inference, dùng similarity giữa input instance và các presentative distributions để chọn LoRA block phù hợp
3. Không cần trainable parameters mới trong quá trình selection

## Đặc điểm kỹ thuật
- **Architecture**: Separate LoRA blocks per task + frozen pre-trained LLM
- **Feature distribution**: Mean vector of pre-trained features per task per layer  
- **Selection**: Dot product / L2 / cosine similarity between input features and stored distributions
- **Multi-module**: CÓ - dùng multiple LoRA blocks, dynamic selection → MULTI-MODULE

## Đánh giá motivation cho Simple Idea
### ❌ KHÔNG PHÙ HỢP (Score: 2/10)

**Lý do loại:**
1. **Vi phạm single-model**: Dùng multiple LoRA blocks (1 per task) + dynamic routing = multi-module architecture
2. **Feature distribution chỉ dùng cho routing**: Paper dùng distribution để SELECT LoRA, không để ANTI-FORGETTING. Simple idea dùng distribution để preserve old cluster geometry
3. **Không có geometry-aware modeling**: Distributions chỉ là mean vectors, không model shape/anisotropy
4. **Paradigm khác**: Task isolation (parameter isolation) vs. single model continual learning

**Điểm tương đồng (nhỏ):**
- Cùng idea dùng feature distribution → nhưng mục đích khác (routing vs. anti-forgetting)
- Cùng nhận thức tầm quan trọng của feature representations
