# specroute v2 (ROUGE-L)

|   | yelp | amazon | mnli | cb | copa | qqp | rte | imdb | sst2 | dbpedia | agnews | yahoo | multirc | boolq | wic |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| yelp | 70.9518 |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
| amazon | 68.9671 | 67.6754 |   |   |   |   |   |   |   |   |   |   |   |   |   |
| mnli | 66.0417 | 64.4759 | 33.8816 |   |   |   |   |   |   |   |   |   |   |   |   |
| cb | 65.6689 | 64.3311 | 31.0263 | 3.5714 |   |   |   |   |   |   |   |   |   |   |   |
| copa | 63.432 | 63.1557 | 13.8816 | 0.0 | 54.0 |   |   |   |   |   |   |   |   |   |   |
| qqp | 62.943 | 62.6623 | 7.8947 | 0.0 | 56.0 | 76.6053 |   |   |   |   |   |   |   |   |   |
| rte | 62.9474 | 62.6316 | 7.6579 | 0.0 | 55.0 | 76.6447 | 6.4982 |   |   |   |   |   |   |   |   |
| imdb | 62.9342 | 61.2807 | 7.5921 | 0.0 | 53.0 | 76.6579 | 6.4982 | 1.8114 |   |   |   |   |   |   |   |
| sst2 | 62.4474 | 49.1754 | 6.1579 | 0.0 | 54.0 | 76.5395 | 6.4982 | 10.4035 | 7.9128 |   |   |   |   |   |   |
| dbpedia | 62.4342 | 49.3026 | 5.3026 | 0.0 | 54.0 | 76.5395 | 6.1372 | 9.693 | 7.2248 | 30.8371 |   |   |   |   |   |
| agnews | 62.4342 | 49.3158 | 5.3026 | 0.0 | 54.0 | 76.5263 | 5.7762 | 9.5219 | 7.1101 | 33.0367 | 36.6687 |   |   |   |   |
| yahoo | 48.7522 | 30.5702 | 4.5395 | 0.0 | 59.0 | 74.6491 | 7.2202 | 6.7456 | 6.1927 | 31.5812 | 36.4073 | 3.6411 |   |   |   |
| multirc | 48.8136 | 30.6404 | 3.9079 | 0.0 | 57.0 | 74.7149 | 7.2202 | 6.6535 | 5.7339 | 31.6773 | 36.3994 | 3.2744 | 46.7616 |   |   |
| boolq | 49.0022 | 31.0614 | 3.7368 | 0.0 | 56.0 | 74.7412 | 7.2202 | 6.614 | 5.7339 | 31.4345 | 36.5626 | 3.2481 | 46.7616 | 51.9776 |   |
| wic | 53.2522 | 33.2939 | 1.7982 | 0.0 | 53.0 | 74.4386 | 18.4116 | 5.7588 | 5.2752 | 32.3429 | 42.3672 | 3.0473 | 47.731 | 51.7023 | 1.7241 |

## Phân tích hiệu năng SpecRoute v2 (ROUGE-L)

Dựa trên bảng kết quả (sử dụng metrics ROUGE-L), ta thấy:

### 1. Hiện tượng Quên (Catastrophic Forgetting)
*   **Mức độ Forget cao:** Phù hợp với nhận định ban đầu, điểm số ROUGE-L tuy cao hơn Exact Match nhưng vẫn sụt giảm mạnh.
    *   `yelp`: **70.95** (init) -> **53.25** (final). Giảm ~17.7 điểm.
    *   `amazon`: **67.67** (init) -> **33.29** (final). Giảm ~34.4 điểm!
    *   `mnli`: **33.88** (init) -> **1.79** (final). Collapse hoàn toàn.
*   `amazon` bị quên nặng hơn `yelp`, có thể do `yelp` là task đầu tiên nên có "ưu thế" trong subspace ban đầu.

### 2. Backward Transfer tích cực trên RTE
*   Một điểm đáng chú ý là `rte` (Task 7) ban đầu chỉ đạt **6.49**, nhưng sau khi học Task 15 (`wic`), điểm số lại tăng lên **18.41**.
*   Điều này cho thấy có sự chia sẻ tri thức (Shared Subspace) có lợi từ các task sau về cho `rte`, mặc dù cơ chế bảo vệ của v2 vẫn còn lỏng lẻo.

### 3. Learning Collapse trên các Task trung bình/nhỏ
*   Các task như `cb`, `imdb`, `sst2`, `wic`, `yahoo` đều có điểm số khởi đầu cực thấp.
*   `cb` (Task 4) đạt **3.57** ban đầu và ngay lập tức về **0.0** sau task 5.
*   `wic` (Task 15) chỉ đạt **1.72**.

### 4. Kết luận
SpecRoute v2 gặp vấn đề nghiêm trọng với các task cùng domain (yelp/amazon/imdb) và các task suy luận (mnli/cb/rte). Mặc dù ROUGE-L cho thấy điểm số "đẹp" hơn Exact Match ở các task phân loại, nhưng xu hướng quên và collapse vẫn rất rõ ràng. Cần áp dụng CPI và OAP (như mô tả trong IDEA_Overall) để cải thiện khả năng phân biệt routing và chia sẻ kiến thức an toàn.
