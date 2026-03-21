# Chapter 1: Cross-Domain Idea Map for Zero-Replay Continual Learning

Bài toán Zero-Replay Continual Learning (CL) với ràng buộc GPM/Orthogonality thực chất là một bài toán về **"Quản lý tài nguyên trong không gian bị giới hạn"**. Dưới đây là các góc nhìn mới mẻ từ các lĩnh vực khác để bạn tìm cảm hứng cho Idea.

---

## 1. Khoa học Não bộ (Neuroscience)

Não bộ học liên tục mà không cần "replay" dữ liệu thô (mặc dù có cơ chế sleep/replay nhưng ở mức trừu tượng).

*   **Synaptic Tagging and Capture (STC)**:
    *   **Problem**: Làm sao để các thay đổi tạm thời (LTP ngắn hạn) trở thành vĩnh viễn (LTP dài hạn) mà không gây nhiễu cho các synapse khác?
    *   **Mechanism**: Mỗi synapse khi được kích hoạt sẽ để lại một "Tag" (đánh dấu hóa học). Sau đó, neuron tiết ra các Protein đặc hiệu (PRP). Chỉ những synapse có "Tag" mới "Capture" được Protein đó để ổn định hóa cấu trúc.
    *   **Idea cho CL**: Thay vì bảo vệ toàn bộ LoRA (r=8), ta chỉ "Tag" những hướng singular vectors quan trọng nhất cho task đó và chỉ bảo vệ (freeze/orthogonalize) những hướng này.
*   **Structural Plasticity**:
    *   **Mechanism**: Não bộ không chỉ thay đổi trọng số mà còn thay đổi kết cấu (tạo/xóa synapse mới). 
    *   **Keyword**: *Neurogenesis*, *Synaptic Pruning*.

**Papers hay ho:**
1.  *Frey, U., & Morris, R. G. (1997). Synaptic tagging and capture.* (Nature)
2.  *Ballarini, F., et al. (2009). Behavioral tagging is a general mechanism of long-term memory formation.* (PNAS)

---

## 2. Xử lý Tín hiệu & Toán học (Signal Processing / Math)

Đây là lĩnh vực cung cấp nền tảng cho GPM và Spectral Routing hiện tại.

*   **Subspace Tracking (Theo dấu không gian con)**:
    *   **Problem**: Dữ liệu thay đổi theo thời gian (streaming), làm sao để cập nhật base (PCA) một cách liên tục mà không cần tính lại toàn bộ covariance matrix?
    *   **Analogy**: GPM hiện tại đang cố gắng "track" không gian null-space.
    *   **Keyword**: *Online PCA*, *Subspace Tracking*, *Projection Approximation Subspace Tracking (PAST)*.
*   **Compressed Sensing (Cảm biến nén)**:
    *   **Mechanism**: Có thể khôi phục một tín hiệu giàu thông tin từ một số lượng rất ít các phép đo (sparse measurements) nếu tín hiệu đó có tính "sparse" trong một domain nào đó.
    *   **Idea cho Routing**: Task identity có thể được xem là một "sparse signal" bị nén trong embedding space. Spectral routing thực chất là một phép giải nén (reconstruction).

**Papers hay ho:**
3.  *CoSO: Continuous Subspace Optimization for Continual Learning (2022)* - Rất gần với SpecRoute, dùng SVD để nén gradients.
4.  *NESS: Null-space Estimated from Small Singular values (2022)* - Dùng singular values nhỏ để ước lượng không gian "trống".
5.  *ROSETA: Robust Online Subspace Estimation and Tracking Algorithm.*

---

## 3. Giao vận & Định tuyến (Logistics / OR)

*   **Districting / Partitioning (Phân khu)**:
    *   **Problem**: Chia một thành phố thành nhiều vùng phục vụ (districts) sao cho chi phí di chuyển thấp nhất nhưng các vùng phải cân bằng (workload balance).
    *   **Analogy**: Việc chia model capacity cho các tasks. Nếu chia quá chặt (GPM chặt), task mới không có "đường đi". Nếu chia lỏng, tasks sẽ lấn sân nhau.
    *   **Idea**: Thay vì chia không gian con (subspaces) cố định, hãy dùng **"Soft Partitioning"** - cho phép các task dùng chung một số "hành lang" (shared subspaces) và chỉ độc quyền ở các "lõi" (core subspaces).

**Papers hay ho:**
6.  *Dynamic Districting for Delivery Problems* (Tạp chí về Operations Research).
7.  *Adaptive Partitioning for Multi-Resource Allocation.*

---

## 4. Các Keyword & Hướng đi "Mới mẻ" (Out-of-the-box)

*   **Keyword 1: Hypernetworks / Metaplasticity**. Thay vì train LoRA, hãy train một mạng nhỏ (Hypernetwork) sinh ra LoRA weights dựa trên task context.
*   **Keyword 2: Reservoir Computing (Echo State Networks)**. Coi T5 backbone là một "bồn chứa" (reservoir) có dynamics phức tạp. Task routing là việc lọc ra đúng "echo" của dữ liệu trong bồn chứa đó.
*   **Keyword 3: Cellular Automata for Neural Growth**. Coi mỗi expert LoRA như một tế bào có thể "sinh trưởng" (tăng rank) hoặc "phân chia" dựa trên nhu cầu dữ liệu.

**Top 3 Papers bạn NÊN đọc ngay:**
8.  **"CoSO: Continuous Subspace Optimization"**: Để xem cách họ tối ưu hóa subspace một cách liên tục thay vì chỉ dùng GPM tĩnh.
9.  **"NESS: Null-space Estimated from Small Singular values"**: Để xem cách tiếp cận đối nghịch với GPM (tìm không gian rác để học).
10. **"A Unified View of Continuous Subspace Tracking"**: Để hiểu sâu về toán học đằng sau việc thay đổi không gian con theo thời gian.

> [!TIP]
> **Observation**: Vấn đề "GPM null-space quá nhỏ" thực chất là bài toán **Fragmented Memory** trong hệ điều hành. Khi file (task) bị xóa/thêm, bộ nhớ bị phân mảnh. Idea có thể là: **Defragmentation** (tối ưu lại các subspaces cũ để gom không gian trống lại cho task mới) — dĩ nhiên phải làm sao để không vi phạm zero-replay.





### Chapter 2: Một số công cụ toán học
Dưới đây là các công cụ toán học có tính cấu trúc cao và đang được ứng dụng rộng rãi trong Machine Learning hiện đại, ngoài Optimal Transport.1. Phương trình vi phân ngẫu nhiên (Stochastic Differential Equations - SDEs) và Neural ODEsCác mô hình học máy truyền thống thường biến đổi dữ liệu qua các tầng rời rạc. SDEs và ODEs chuyển đổi quá trình này thành các quỹ đạo liên tục theo thời gian.Cơ sở toán học: Giải tích Itô (Itô calculus), phương trình đạo hàm riêng (PDEs).Ứng dụng thực tiễn: * Score-based Generative Models (Mô hình khuếch tán): Quá trình phá hủy và tạo sinh dữ liệu (như hình ảnh, âm thanh) được mô hình hóa bằng các phương trình vi phân ngẫu nhiên dạng:$$dx = f(x, t)dt + g(t)dw$$trong đó $w$ là quá trình Wiener (chuyển động Brown chuẩn).Neural ODEs: Mô hình hóa chuỗi thời gian liên tục hoặc dữ liệu có khoảng cách lấy mẫu không đều, giảm thiểu bộ nhớ do không cần lưu trữ các trạng thái trung gian trong quá trình lan truyền ngược (backpropagation).2. Hình học vi phân (Differential Geometry) và Nhóm Lie (Lie Groups)Công cụ này giải quyết vấn đề học trên các miền dữ liệu phi Euclid (đồ thị, đa tạp) và yêu cầu tính bất biến/hiệp biến (invariance/equivariance) đối với các phép biến đổi không gian.Cơ sở toán học: Đa tạp Riemann (Riemannian manifolds), Đại số Lie, lý thuyết biểu diễn nhóm.Ứng dụng thực tiễn:Geometric Deep Learning: Thiết kế các mạng nơ-ron có khả năng bảo toàn tính đối xứng của dữ liệu.Khoa học tự nhiên: Dự đoán cấu trúc protein (như AlphaFold) hoặc mô phỏng động học phân tử, nơi các phép xoay và tịnh tiến (thuộc nhóm $SE(3)$) không làm thay đổi bản chất của hệ thống vật lý.Tối ưu hóa đa tạp: Giới hạn không gian tìm kiếm của thuật toán tối ưu hóa trên các bề mặt phi tuyến, giúp hội tụ nhanh hơn đối với một số hàm mục tiêu cụ thể.3. Phân tích dữ liệu tô pô (Topological Data Analysis - TDA)TDA cung cấp phương pháp trích xuất các đặc trưng hình học toàn cục của dữ liệu (như số lượng thành phần liên thông, lỗ hổng nhiều chiều) mà không bị ảnh hưởng bởi các biến dạng nhỏ hoặc nhiễu cục bộ.Cơ sở toán học: Đại số đồng điều (Algebraic Topology), đặc biệt là Persistent Homology.Ứng dụng thực tiễn:Tính toán sự thay đổi của các cấu trúc tô pô ở các thang đo khác nhau (filtration), tạo ra các biểu đồ (persistence diagrams) dùng làm đặc trưng đầu vào cho các mô hình học máy.Phân tích mạng lưới phức tạp, dữ liệu sinh học (như biểu hiện gen) hoặc hình ảnh y tế, nơi cấu trúc không gian có tính chất quyết định nhưng dữ liệu thu thập bị nhiễu cao.4. Hình học thông tin (Information Geometry)Công cụ này nghiên cứu không gian của các phân phối xác suất bằng cách sử dụng các khái niệm từ hình học vi phân.Cơ sở toán học: Đa tạp thống kê, ma trận thông tin Fisher, khoảng cách Kullback-Leibler.Ứng dụng thực tiễn:Natural Gradient Descent: Thay vì cập nhật trọng số dựa trên khoảng cách Euclid trong không gian tham số, thuật toán sử dụng ma trận thông tin Fisher để đo lường khoảng cách trong không gian phân phối xác suất. Điều này giúp quỹ đạo tối ưu hóa đi theo con đường dốc nhất thực sự của hàm suy hao, độc lập với cách tham số hóa mô hình.