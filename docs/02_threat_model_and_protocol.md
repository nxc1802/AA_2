# 02. Mô Hình Mối Đe Dọa & Giao Thức Đánh Giá (Threat Model & Protocol SSOT)

> **Dự án:** AA (Sparse Adversarial Attack Benchmark)  
> **Phương pháp trung tâm:** CASA  
> **Trạng thái:** Authoritative Single Source of Truth (Frozen Protocol)  
> **Mục tiêu:** Định nghĩa chuẩn toán học về không gian đe dọa, chuẩn spatial $L_0$, mẫu số đánh giá, các chỉ số đo lường hiệu năng/chất lượng ảnh, quy tắc hạch toán chi phí tính toán và chính sách thực nghiệm.

---

## 1. Mô Hình Mối Đe Dọa Toán Học ($L_0$ Threat Model)

### 1.1 Không gian đầu vào và bộ phân loại
Cho bộ phân loại ảnh sâu:
$$f: [0, 1]^{3 \times H \times W} \to \mathbb{R}^C$$
nhận đầu vào ảnh $x$ với 3 kênh màu RGB được chuẩn hóa trong siêu hộp compact $[0, 1]^{3 \times H \times W}$, đưa ra vector logit $z(x) = f(x)$ trên $C$ lớp nhãn. Nhãn dự đoán là:
$$\hat{y}(x) = \arg\max_{c \in \{1, \dots, C\}} z_c(x)$$

### 1.2 Ràng buộc nhiễu đối nghịch và Chuẩn giả Thưa thớt Không gian
Một đòn tấn công đối nghịch tìm kiếm vector nhiễu $\delta \in \mathbb{R}^{3 \times H \times W}$ sao cho ảnh đối nghịch $x' = x + \delta$ thỏa mãn hai điều kiện nghiêm ngặt:

1. **Ràng buộc hộp hợp lệ (Box Constraint):**
   $$x' = x + \delta \in [0, 1]^{3 \times H \times W}$$
   Mọi giá trị pixel sau khi can thiệp bắt buộc phải nằm trong miền giá trị ảnh hợp lệ, không xảy ra hiện tượng tràn giá trị.

2. **Ràng buộc ngân sách thưa thớt không gian (Spatial $L_0$ Constraint):**
   $$\|\delta\|_{0, \text{spatial}} \le K$$

   **Định nghĩa chuẩn giả Spatial $L_0$:** Một tọa độ không gian $(h, w)$ được tính là bị can thiệp nếu tồn tại ít nhất một kênh màu có độ lệch tuyệt đối vượt qua ngưỡng sai số số học $\tau = 10^{-5}$:
   $$\|\delta\|_{0, \text{spatial}} = \sum_{h=1}^H \sum_{w=1}^W \mathbb{I}\left( \max_{c \in \{R, G, B\}} |x'_{c, h, w} - x_{c, h, w}| > \tau \right)$$

   Ngưỡng $\tau = 10^{-5}$ được thiết lập nhằm triệt tiêu các sai số dấu phẩy động (floating-point precision noise) khi tối ưu hóa gradient trên GPU.

### 1.3 Tập hỗ trợ không gian (Spatial Support)
Tập hỗ trợ $S$ được định nghĩa là tập các chỉ số tọa độ phẳng bị can thiệp:
$$S = \left\{ i = (h - 1)W + w \;\middle|\; \max_{c} |\delta_{c, h, w}| > \tau \right\} \subseteq \{0, 1, \dots, H \times W - 1\}$$
với lực lượng $|S| \le K$. Tất cả 3 kênh màu $\{R, G, B\}$ tại mỗi tọa độ $i \in S$ đều được phép thay đổi đồng thời.

---

## 2. Các Mốc Ngân Sách Chuẩn Thức (Canonical Budget Grid)

Bộ tiêu chuẩn đánh giá cố định 7 mốc ngân sách pixel:
$$K \in \{1, 2, 4, 8, 16, 32, 64\}$$

* **Vùng cực thưa (Extreme Sparsity - $K \in \{1, 2\}$):** Can thiệp 1 đến 2 điểm ảnh trên toàn ảnh (chiếm $0.098\% \to 0.195\%$ diện tích ảnh CIFAR 32×32). Đây là vùng thử thách cao nhất cho các phương pháp tìm kiếm tập hỗ trợ.
* **Vùng trung bình (Medium Sparsity - $K \in \{4, 8\}$):** Vùng chuyển tiếp, nơi tương tác liên minh (synergy) giữa các cụm pixel thể hiện rõ nét nhất.
* **Vùng thưa mở rộng (Expanded Sparsity - $K \in \{16, 32, 64\}$):** Đánh giá khả năng bão hòa tấn công của giải thuật trước khi đạt giới hạn bẻ gãy hoàn toàn.

---

## 3. Chính Sách Mẫu Số Đánh Giá & Định Nghĩa Chỉ Số (Metrics Specification)

### 3.1 Quy tắc mẫu số chuẩn: Mẫu phân loại đúng ban đầu (Clean-Correct Denominator)
Để đảm bảo tính nghiêm cẩn khoa học và loại trừ sự thiên lệch do các mẫu ảnh sạch vốn đã bị mô hình phân loại sai ($f(x_i) \neq y_i$), toàn bộ các chỉ số tấn công và phòng thủ đều phải sử dụng tập mẫu sạch phân loại đúng làm mẫu số:

$$\mathcal{D}_{\text{clean-correct}} = \left\{ (x_i, y_i) \in \mathcal{D}_{\text{test}} \;\middle|\; \arg\max f(x_i) = y_i \right\}$$
với kích thước:
$$N_{\text{clean}} = |\mathcal{D}_{\text{clean-correct}}| = \sum_{i=1}^N \mathbb{I}\left( \arg\max f(x_i) = y_i \right)$$

### 3.2 Tỷ lệ tấn công thành công có điều kiện (Conditional Attack Success Rate - ASR@$K$)
Mẫu $x_i$ được coi là bị tấn công thành công nếu thỏa mãn đồng thời: (1) mẫu sạch được dự đoán đúng, (2) mẫu đối nghịch bị dự đoán sai nhãn, và (3) số pixel can thiệp không vượt quá ngân sách $K$:

$$\text{Success}(x_i; K) = \mathbb{I}\left( \arg\max f(x_i) = y_i \;\land\; \arg\max f(x_i + \delta_i) \neq y_i \;\land\; \|\delta_i\|_{0, \text{spatial}} \le K \right)$$

Tỷ lệ ASR@$K$ được tính theo công thức:
$$\text{ASR}@K = \frac{\sum_{i=1}^N \text{Success}(x_i; K)}{N_{\text{clean}}} \times 100\%$$

### 3.3 Độ chính xác bền vững có điều kiện (Conditional Robust Accuracy - CRA@$K$)
$$\text{CRA}@K = 100\% - \text{ASR}@K = \frac{\sum_{i=1}^N \mathbb{I}\left( \arg\max f(x_i) = y_i \;\land\; \arg\max f(x_i + \delta_i) = y_i \right)}{N_{\text{clean}}} \times 100\%$$

> **Bất biến toán học bắt buộc (Invariance Assertion):**
> $$\text{ASR}@K + \text{CRA}@K \equiv 100.00\% \quad (\forall K)$$
> Mọi hệ thống kiểm thử tự động đều phải kiểm tra tính bất biến này.

### 3.4 Độ chính xác bền vững toàn tập (Full-Set Robust Accuracy - RA@$K$)
Tỷ lệ mẫu được phân loại chính xác trên toàn bộ tập dữ liệu (bao gồm cả các mẫu sạch bị sai):
$$\text{RA}@K = \text{Clean Accuracy} \times \frac{\text{CRA}@K}{100\%} = \frac{\sum_{i=1}^N \mathbb{I}\left( \arg\max f(x_i + \delta_i) = y_i \right)}{N_{\text{total}}} \times 100\%$$

---

## 4. Các Chỉ Số Chất Lượng Nhiễu & Ảnh Đối Nghịch

Bên cạnh tỷ lệ bẻ gãy mô hình, mỗi thực nghiệm bắt buộc phải ghi nhận các chỉ số về biên độ nhiễu và độ suy giảm thị giác:

1. **Chuẩn đạt được thực tế (Achieved $L_0$):**
   $$\overline{L}_0 = \frac{1}{|\text{Success}|} \sum_{x \in \text{Success}} \|\delta\|_{0, \text{spatial}}$$
   *Đối với các thuật toán có cơ chế nén như CASA (Drop-and-Repair), $\overline{L}_0$ thường nhỏ hơn đáng kể so với trần $K$.*

2. **Độ lệch Euclidean ($L_2$ Distortion):**
   $$\|\delta\|_2 = \sqrt{\sum_{c=1}^3 \sum_{h=1}^H \sum_{w=1}^W \delta_{c, h, w}^2}$$

3. **Độ lệch tối đa ($L_\infty$ Distortion):**
   $$\|\delta\|_\infty = \max_{c, h, w} |\delta_{c, h, w}|$$

4. **Tỷ số tín hiệu cực đại trên nhiễu (Peak Signal-to-Noise Ratio - PSNR):**
   $$\text{PSNR}(x, x') = 10 \cdot \log_{10}\left( \frac{\text{MAX}_I^2}{\text{MSE}(x, x')} \right) \quad (\text{với } \text{MAX}_I = 1.0)$$

5. **Chỉ số tương đồng cấu trúc (Structural Similarity Index - SSIM):**
   Đo lường mức độ suy giảm cấu trúc cảm thụ thị giác theo thang điểm $[0, 1]$.

---

## 5. Chuẩn Hóa Hạch Toán Chi Phí Tính Toán (Query & Cost Accounting)

Để đảm bảo tính công bằng khoa học giữa các họ giải thuật khác nhau, dự án bác bỏ việc chỉ so sánh thời gian chạy thực tế (wall-clock time) do phụ thuộc vào mức độ tối ưu hóa code và phần cứng. Chi phí tính toán được phân rã thành các chỉ số độc lập:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                      HẠCH TOÁN CHI PHÍ TÍNH TOÁN                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Zero-Order Queries (Hộp đen):                                           │
│    • Số lần gọi mô hình chỉ tính lan truyền xuôi f(x), không lấy đạo hàm.  │
│    • Áp dụng cho: Sparse-RS, CornerSearch.                                  │
│                                                                             │
│ 2. First-Order Evaluations (Hộp trắng):                                     │
│    • Forward Passes (F): Số lần lan truyền xuôi qua mạng (inference/loss).  │
│    • Backward Passes (B): Số lần lan truyền ngược qua mạng (vector-Jacobian)│
│                                                                             │
│ 3. FLOP-Equivalent Model Evaluations:                                       │
│    • Chi phí chuẩn hóa quy đổi: FLOP-Cost ≈ F + 2 × B                       │
│    (Vì 1 lần tính backward tiêu tốn năng lượng tính toán xấp xỉ 2 lần forward)│
└─────────────────────────────────────────────────────────────────────────────┘
```

Tuyệt đối **không gộp** số lần tính gradient hộp trắng vào cùng trường `queries` với các đòn tấn công hộp đen.

---

## 6. Chính Sách Tinh Chỉnh Siêu Tham Số, Seed & Thống Kê

### 6.1 Chính sách tinh chỉnh siêu tham số (Hyperparameter Tuning Policy)
- **Cấm hoàn toàn việc tune tham số trên Test Set:** Mọi siêu tham số của CASA (bước lặp $T_{\text{outer}}$, $T_{\text{inner}}$, learning rate $\alpha$, bán kính NMS $r$, hệ số nhân candidate pool $M$) đều được cố định trên tập phát triển (Development split / Validation subset gồm 1.000 mẫu) trước khi đóng băng chạy trên tập kiểm thử chính thức (Test Set 10.000 mẫu).
- **Tham số của đối thủ (Baselines):** Sử dụng cấu hình khuyến nghị chính thức của các tác giả từ bài báo gốc hoặc repository chính thức đã thẩm định.

### 6.2 Chính sách Random Seed & Tính Ổn Định
- **Seed chuẩn thức (Canonical Seed):** `seed = 42` được dùng cho toàn bộ các thực nghiệm chính thức.
- **Tập đa seed (Multi-seed Set):** `{42, 123, 999}` được sử dụng bắt buộc cho các thí nghiệm đánh giá độ ổn định phương sai (H6 / Ablation).
- **RNG Determinism:** Khởi tạo seed đồng thời cho `torch.manual_seed`, `torch.cuda.manual_seed_all`, `np.random.seed` và thiết lập `torch.backends.cudnn.deterministic = True`. Trạng thái sinh số ngẫu nhiên cho từng mẫu ảnh được gắn với:
  $$\text{Seed}_{\text{sample}} = \text{Seed}_{\text{global}} + \text{Sample\_Index}$$

### 6.3 Chính sách Khoảng Tin Cậy Thống Kê (Statistical Confidence Policy)
Mọi kết quả ASR@$K$ trên tập kiểm thử đều phải tính kèm **Khoảng tin cậy Wilson Score 95%** (Wilson Score Interval cho phân phối nhị thức):

$$\text{CI}_{95\%} = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$
với $z = 1.96$, $\hat{p} = \frac{\text{Success}}{N_{\text{clean}}}$, và $n = N_{\text{clean}}$.

Đối với các so sánh trực diện giữa CASA và baseline trên cùng một tập mẫu, sử dụng **Kiểm định McNemar (McNemar's Test)** cho dữ liệu ghép cặp (paired nominal data) để kết luận ý nghĩa thống kê ($p < 0.01$).
