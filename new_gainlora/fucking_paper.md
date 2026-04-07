# Tiêu đề: Statistical Routing Theory and Capacity Analysis for LoRA-based Continual Learning

## 1. Abstract
Trong Học liên tục (Continual Learning - CL), các phương pháp dựa trên Low-Rank Adaptation (LoRA) đang giải quyết bài toán định tuyến (routing) và nhiễu (interference) thông qua hai hướng tiếp cận chính: không gian rỗng trực giao (như O-LoRA, InfLoRA) hoặc định tuyến có học (như SAPT, GainLoRA). Tuy nhiên, các phương pháp này đối mặt với sự cạn kiệt không gian biểu diễn nhanh chóng hoặc rủi ro quên thảm khốc (catastrophic forgetting) trong chính mạng định tuyến. Nghiên cứu này tiếp cận bài toán từ góc độ hình học không gian nhúng. Dựa trên phát hiện (Ethayarajh, 2019) rằng các vector nhúng của LLMs có tính dị hướng cao (anisotropic), chúng tôi lập luận rằng sự thất bại của các phương pháp trên xuất phát từ việc áp đặt các phép đo lường và biến đổi trên một không gian bị méo lệch. Bằng cách áp dụng phép biến đổi ZCA Whitening để chuyển đổi phân phối dị hướng thành phân phối chuẩn đẳng hướng (isotropic), chúng tôi chứng minh rằng khoảng cách Euclidean (L2) nguyên bản sẽ hoạt động như một bộ định tuyến Mahalanobis tối ưu mà không cần huấn luyện thêm. Dựa trên không gian đã chuẩn hóa, chúng tôi đề xuất khung phân tích Tỷ lệ Tham gia (Participation Ratio - PaR) để định lượng chính xác điểm cạn kiệt dung lượng hệ thống. Tính hiệu quả và độ khái quát của phương pháp được chứng minh thực nghiệm trên các benchmark tiêu chuẩn, áp dụng trực tiếp làm nền tảng (backbone) cho các cấu trúc GainLoRA và MoDE.

## 2. Introduction

**Sự tiến hóa và các điểm nghẽn của LoRA-CL hiện tại**
Để giải quyết bài toán suy giảm hiệu suất trong Học liên tục, cấu trúc LoRA với các bộ điều hợp (adapter) độc lập cho từng tác vụ được sử dụng rộng rãi. Ban đầu, các nghiên cứu tập trung vào cơ chế trực giao hóa để ngăn chặn nhiễu chéo. O-LoRA (2024) khởi xướng phương pháp học các không gian con trực giao, sau đó InfLoRA (2024) phát triển tiếp bằng cách ép các tác vụ mới chiếu hoàn toàn vào không gian rỗng (null-space) của các tác vụ cũ. Song song đó, để giải quyết bài toán chọn đúng adapter khi suy diễn (task-agnostic inference), các cấu trúc như SAPT và gần đây là GainLoRA (2025) sử dụng các cơ chế cổng (gates) hoặc mạng định tuyến (routers) có thể học được (learnable).

Tuy nhiên, quỹ đạo tiếp cận này bộc lộ các vấn đề cơ bản. Hướng đi của InfLoRA triệt tiêu đặc trưng hữu ích khi các tác vụ có sự chồng lấn, dẫn đến cạn kiệt không gian biểu diễn cực nhanh đối với các chuỗi tác vụ cùng miền (same-domain). Trong khi đó, hướng tiếp cận của GainLoRA đặt gánh nặng lên một mạng định tuyến phụ trợ, yêu cầu mạng này phải liên tục cập nhật, khiến bản thân bộ định tuyến trở thành điểm yếu dễ bị trôi lệch (drift) và chịu ảnh hưởng của catastrophic forgetting.

**Cơ sở hình học và Giải pháp đề xuất**
Sự bế tắc của các phương pháp trên xuất phát từ việc bỏ qua đặc tính hình học cốt lõi của dữ liệu đầu ra từ LLM. Ethayarajh (2019) đã chứng minh rằng các vector nhúng ngữ cảnh (contextual embeddings) không phân bổ đồng đều mà tạo thành một cấu trúc nón hẹp có tính dị hướng cực đại (anisotropic hypersphere). Việc áp dụng khoảng cách L2 nguyên bản, trực giao hóa (như InfLoRA), hoặc dùng mạng nơ-ron để chia cắt một không gian vốn đã bị biến dạng sẽ dẫn đến sai số lớn.

Lý thuyết thống kê đa biến chỉ ra rằng, đối với một phân phối dị hướng (dạng ellipsoid), phép biến đổi Whitening có khả năng chuyển đổi nó thành một phân phối đẳng hướng (dạng hình cầu chuẩn hóa). Từ cơ sở toán học này, chúng tôi lập luận rằng: thay vì xây dựng các cơ chế định tuyến phức tạp hay ép buộc không gian rỗng, giải pháp cốt lõi là chuẩn hóa lại không gian nhúng. 

Khi dữ liệu được đi qua lớp ZCA Whitening, khoảng cách L2 cơ bản trở nên tương đương toán học với khoảng cách Pooled Mahalanobis trong không gian gốc. Điều này cho phép một cơ chế Nearest Centroid hoàn toàn tĩnh đạt độ chính xác định tuyến tiệm cận các phương pháp có học (learned routers) mà không chịu bất kỳ rủi ro forgetting nào. Hơn nữa, trên không gian đẳng hướng này, dung lượng biểu diễn thực tế của các adapter có thể được định lượng chính xác thông qua Tỷ lệ Tham gia (Participation Ratio - PaR), thay vì phụ thuộc vào hạng (rank) danh nghĩa.

**Đóng góp của bài báo (Contributions):**
1. Đề xuất Whitened L2 Routing: Một phương pháp định tuyến tĩnh, can thiệp tối thiểu, khắc phục hoàn toàn rủi ro trôi lệch của các learned routers bằng cách khai thác cấu trúc dị hướng của LLMs.
2. Thiết lập khung phân tích dung lượng LoRA trong CL dựa trên PaR, cung cấp công thức định lượng điểm kiệt quệ thứ hạng (Rank Exhaustion) để giải thích giới hạn của các cấu trúc dựa trên null-space.
3. Chứng minh tính khái quát của phương pháp: Khung lý thuyết và phương pháp định tuyến được tích hợp và chứng minh hiệu năng vượt trội trên các benchmark tiêu chuẩn, áp dụng trực tiếp lên các kiến trúc SOTA hiện hành bao gồm GainLoRA và MoDE.

## 3. Related Work

*(Phần này sắp xếp lại các nghiên cứu theo đúng mạch logic của Introduction, củng cố thêm tài liệu tham khảo)*

**3.1. Continual Learning with LoRA and Routing Mechanisms**
Trình bày chuỗi phát triển: O-LoRA $\rightarrow$ InfLoRA (hướng trực giao/null-space) và SAPT $\rightarrow$ GainLoRA $\rightarrow$ MoDE (hướng router/gate có học). Phân tích rõ nhược điểm vật lý: cạn kiệt số chiều hiệu dụng và rủi ro quên của router.

**3.2. LLM Embedding Geometry and Whitening**
Trích dẫn Ethayarajh (2019) về Representation Degeneration và tính dị hướng. Liên kết với các nghiên cứu về Whitening trong xử lý tín hiệu thống kê (ví dụ: Kessy et al., 2018) để khẳng định cơ sở toán học của phép chuyển đổi từ không gian dị hướng sang đẳng hướng. Sự giao thoa giữa hai mảng tài liệu này chính là nền tảng hình thành phương pháp đề xuất.