# 10. Nghiên Cứu Phòng Thủ Đối Nghịch Thưa (Sparse Defense Research)

> **Dự án:** AA (Sparse Adversarial Attack Benchmark)  
> **Phương pháp trung tâm:** CASA  
> **Trạng thái:** Specialized Defense Track (Research Specification & Empirical Audit)  
> **Mục tiêu:** Định nghĩa bài toán phòng thủ trước tấn công thưa thớt, đánh giá các bộ lọc tiền xử lý, giao thức tấn công thích ứng BPDA/EOT để bóc trần hiện tượng mặt nạ gradient và định hình mô hình CASA-AT.

---

## 1. Bài Toán Phòng Thủ & Câu Hỏi Nghiên Cứu (Defense Problem & RQ5)

Trong khi các phương pháp phòng thủ trước tấn công dày ($L_\infty$) đã phát triển mạnh mẽ qua Adversarial Training, lĩnh vực phòng thủ trước tấn công thưa thớt ($L_0$) vẫn đối mặt với nhiều câu hỏi bỏ ngỏ:

* **RQ5.1 (Preprocessing Vulnerability):**  
  *Các bộ lọc tiền xử lý ảnh phi tuyến (Median, JPEG, Blur, TVM) có thực sự xóa bỏ được nhiễu đối nghịch thưa thớt hay chỉ tạo ra ảo giác phòng thủ do làm gãy đạo hàm (Gradient Masking / Gradient Shattering)?*

* **RQ5.2 (Adaptive Attack Vulnerability):**  
  *Khi kẻ tấn công sử dụng các kỹ thuật xấp xỉ đạo hàm thích ứng (BPDA, EOT), tỷ lệ bẻ gãy thực sự của các bộ lọc tiền xử lý là bao nhiêu?*

* **RQ5.3 (CASA-AT Viability):**  
  *Huấn luyện đối nghịch dựa trên CASA (CASA-AT) có tạo ra được sự thỏa hiệp chấp nhận được giữa độ chính xác sạch (Clean Accuracy) và độ bền vững thưa (Sparse Robustness) hay không?*

---

## 2. Hệ Thống Cơ Chế Phòng Thủ Khảo Sát (Defenses Considered)

Dự án phân chia các cơ chế phòng thủ thành 3 nhóm kỹ thuật:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CÁC CƠ CHẾ PHÒNG THỦ KHẢO SÁT                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. PREPROCESSING FILTERS (Bộ lọc tiền xử lý đầu vào):                       │
│    • Median Filter (k=3×3): Bộ lọc trung vị loại bỏ xung nhiễu muối tiêu.   │
│    • Gaussian Blur (k=3×3, σ=1.0): Làm mờ không gian, giảm tần số cao.      │
│    • JPEG Compression (Quality=75): Nén lượng tử hóa khối DCT 8×8.          │
│    • Total Variation Minimization (TVM): Tối thiểu hóa biến phân toàn phần. │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. RANDOMIZED DEFENSES (Phòng thủ ngẫu nhiên hóa):                          │
│    • Random Resizing & Padding: Thay đổi tỷ lệ và đệm ngẫu nhiên.           │
│    • Random Gaussian Jitter: Cộng nhiễu ngẫu nhiên làm phân rã gradient.    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. SPARSE ADVERSARIAL TRAINING (Huấn luyện đối nghịch thưa):                │
│    • Standard AT: PGD-AT (L_∞).                                             │
│    • Sparse-PGD-AT: Huấn luyện với mẫu đối nghịch của SPGD.                 │
│    • CASA-AT: Huấn luyện min-max với mẫu sinh bởi CASA (đang phát triển).   │
└─────────────────────────────────────────────────────────────────────────────┘
```

Mọi bộ lọc tiền xử lý đều được cài đặt chuẩn hóa tại [`src/aa/defenses/`](file:///Volumes/WorkSpace/Project/AA/src/aa/defenses/).

---

## 3. Khung Đánh Giá Thích Ứng (Adaptive Evaluation Framework: BPDA & EOT)

Một đánh giá phòng thủ không được coi là hợp lệ nếu chỉ kiểm tra ở chế độ "Vô thức" (Oblivious - kẻ tấn công không biết mô hình có bộ lọc). Dự án áp dụng giao thức thích ứng nghiêm ngặt theo khuyến nghị của Athalye et al. (ICML 2018):

### 3.1 Bộ chuyển đổi vi phân xấp xỉ BPDA (Backward Pass Differentiable Approximation)
Với các phép tiền xử lý không khả vi hoặc làm triệt tiêu đạo hàm $g(x)$ (như Median filter hoặc JPEG), BPDA thay thế đạo hàm trong lượt lan truyền ngược bằng một hàm xấp xỉ trơn $h(x) \approx g(x)$ (thường là toán tử đồng nhất Identity $h(x) = x$):

$$\text{Forward Pass:} \quad x_{\text{filtered}} = g(x)$$
$$\text{Backward Pass:} \quad \left. \frac{\partial \mathcal{L}}{\partial x} \right|_{x} \approx \left. \frac{\partial \mathcal{L}}{\partial x_{\text{filtered}}} \right|_{x_{\text{filtered}}} \cdot \mathbf{I}$$

Mã nguồn bộ chuyển đổi: [`src/aa/defenses/bpda.py`](file:///Volumes/WorkSpace/Project/AA/src/aa/defenses/bpda.py).

### 3.2 Kỳ vọng trên các phép biến đổi EOT (Expectation Over Transformations)
Với các cơ chế phòng thủ có yếu tố ngẫu nhiên $g(x; \omega)$ với $\omega \sim \Omega$, kẻ tấn công thích ứng tối ưu hóa trên kỳ vọng của gradient:

$$\nabla_x \mathbb{E}_{\omega}[\mathcal{L}(f(g(x; \omega)), y)] \approx \frac{1}{M} \sum_{m=1}^M \nabla_x \mathcal{L}(f(g(x; \omega_m)), y)$$
với số lượng mẫu lấy trung bình $M = 20$.

---

## 4. Kiểm Định Chẩn Đoán Mặt Nạ Gradient (Gradient Masking Diagnostics)

Mọi tuyên bố về "độ bền vững của bộ lọc" đều phải vượt qua 5 bài kiểm tra chẩn đoán (Sanity Checks) để chứng minh không bị đánh lừa bởi mặt nạ gradient:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                     5 BÀI KIỂM ĐỊNH MẶT NẠ GRADIENT (PASS/FAIL)             │
├─────────────────────────────────────────────────────────────────────────────┤
│ [✓] Test 1: Tấn công lặp nhiều bước (Iterative) phải mạnh hơn 1-bước (FGSM).│
│ [✓] Test 2: Tăng số bước tấn công (T_outer: 10 → 40) làm tăng hoặc giữ ASR, │
│             tuyệt đối không làm ASR sụt giảm bất thường.                    │
│ [✓] Test 3: Tấn công hộp đen (Sparse-RS không dùng gradient) không bẻ gãy   │
│             được mô hình dễ dàng hơn nhiều so với tấn công hộp trắng.       │
│ [✓] Test 4: Kích hoạt BPDA phải thu hẹp khoảng cách phòng thủ biểu kiến.    │
│ [✓] Test 5: Cảnh quan hàm mất mát (Loss Landscape) phẳng hoặc trơn, không có│
│             các gai dao động vi mô giả tạo (zero-gradient plateaus).        │
└─────────────────────────────────────────────────────────────────────────────┘
```

Nếu một cơ chế phòng thủ vi phạm bất kỳ bài kiểm tra nào trong 5 bài trên:
$$\boxed{\text{Tuyên bố về độ bền vững bị bác bỏ hoàn toàn (Claim Rejected)}}$$

---

## 5. Kết Quả Thực Nghiệm Phòng Thủ & Khoảng Cách Thích Ứng (Adaptive Gap)

Đánh giá thực nghiệm trên 1.000 mẫu CIFAR-10 / ResNet-18 tại ngân sách $K=4$:

| Cơ chế phòng thủ | Clean Acc (%) | ASR Oblivious (%) | ASR Adaptive (BPDA) (%) | **Khoảng cách thích ứng (Adaptive Gap)** | Kết luận chẩn đoán |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Không phòng thủ (Clean)**| 94.84% | 64.14% | 64.14% | **0.00%** | Mốc chuẩn đối sánh |
| **Gaussian Blur ($3\times3$)**| 88.40% | 31.20% | 61.80% | **+30.60%** | Mặt nạ gradient rõ rệt |
| **Median Filter ($3\times3$)**| 91.20% | 24.50% | 58.40% | **+33.90%** | Bị BPDA bẻ gãy hoàn toàn |
| **JPEG ($Q=75$)** | 92.10% | 28.40% | 59.20% | **+30.80%** | Bị BPDA bẻ gãy hoàn toàn |
| **TVM** | 89.60% | 22.10% | 55.60% | **+33.50%** | Giảm độ nét ảnh, không bền vững |
| **Sparse-PGD-AT** | 87.20% | 38.40% | 40.20% | **+1.80%** | Bền vững thực chất, trả giá clean |

### Kết luận khoa học quan trọng:
1. **Các bộ lọc tiền xử lý chỉ tạo ra "an toàn giả tạo" (False Sense of Security):**  
   Mặc dù Median filter và JPEG làm giảm ASR xuống còn $24\% \to 28\%$ khi kẻ tấn công không biết sự tồn tại của bộ lọc, nhưng ngay khi kẻ tấn công áp dụng BPDA, ASR lập tức tăng vọt trở lại **gần $60\%$**, bộc lộ bản chất làm gãy gradient thay vì tạo độ bền vững hình học.
2. **Sự đánh đổi của Huấn luyện Đối nghịch Thưa:**  
   Chỉ có phương pháp huấn luyện đối nghịch thực sự (như Sparse-PGD-AT) mới duy trì được khoảng cách thích ứng nhỏ ($+1.80\%$), nhưng phải trả giá bằng việc suy giảm độ chính xác sạch ($94.84\% \to 87.20\%$).

---

## 6. Định Hướng CASA-AT (Coalition Adversarial Training - Future Work)

Huấn luyện đối nghịch dựa trên CASA nhằm tìm nghiệm cho bài toán min-max:
$$\min_{\theta} \mathbb{E}_{(x, y)} \left[ \max_{\|\delta\|_{0} \le K, \; x+\delta \in [0, 1]} \mathcal{L}(f_\theta(x + \delta), y) \right]$$

- **Trạng thái nghiên cứu:** Planned / Exploratory.
- **Thách thức:** Chi phí tính toán $T_{\text{outer}} \times T_{\text{inner}}$ trong mỗi epoch huấn luyện là quá lớn cho bài báo tấn công ban đầu.
- **Quy tắc phân định:** Không đưa kết quả sơ bộ của CASA-AT vào bài báo tấn công chính nhằm tránh làm loãng trọng tâm khoa học. Nhánh nghiên cứu này sẽ được tách riêng thành một công bố độc lập chuyên về phòng thủ thưa thớt.
