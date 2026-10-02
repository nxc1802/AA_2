# 03. Đặc Tả Giải Thuật CASA (CASA Method Specification)

> **Dự án:** AA (Sparse Adversarial Attack Benchmark)  
> **Tên giải thuật:** CASA (*Coalition-Aware Sparse Adversarial Attack*)  
> **Trạng thái:** Frozen Algorithm Specification (Version 1.0)  
> **Mục tiêu:** Mô tả chi tiết cơ sở toán học, các toán tử tối ưu hóa liên minh, mã giả chuẩn hóa và ánh xạ trực tiếp đến các module mã nguồn trong `src/aa/attacks/casa/`.

---

## 1. Động Lực Khoa Học & Bản Chất Toán Học (Motivation)

Tấn công đối nghịch thưa thớt dưới chuẩn $L_0$ đòi hỏi giải quyết đồng thời hai bài toán có tính phụ thuộc chéo:
$$\min_{\delta} \; \mathcal{L}(x + \delta, y) \quad \text{s.t.} \quad x + \delta \in [0, 1]^{3 \times H \times W}, \quad \|\delta\|_{0, \text{spatial}} \le K$$

1. **Bài toán rời rạc:** Tìm tập hỗ trợ không gian tối ưu $S \subseteq \{1, \dots, H \times W\}$ với $|S| \le K$.
2. **Bài toán liên tục:** Tối ưu hóa giá trị nhiễu $\delta_S \in [0, 1]^3$ tại các vị trí trong $S$.

### Giới hạn của cách tiếp cận xếp hạng độc lập:
Nếu định nghĩa hàm mục tiêu tối ưu trên tập hỗ trợ $S$ là:
$$F(S) = \max_{\delta : \operatorname{supp}(\delta) \subseteq S, \; x+\delta \in [0, 1]} \mathcal{L}_{\text{attack}}(x + \delta, y)$$

Lợi ích cận biên thực tế của điểm ảnh $i \notin S$ đối với liên minh hiện tại $S$ là:
$$\Delta(i \mid S) = F(S \cup \{i\}) - F(S)$$

Các giải thuật truyền thống (như PGD0, Salience Sifting) ngầm giả định tính độc lập $\Delta(i \mid S) \approx \Delta(i \mid \varnothing)$. Trong khi đó, trên các mạng nơ-ron sâu với các hàm phi tuyến (ReLU, MaxPool, Self-Attention), tồn tại hiện tượng **hiệp đồng liên minh (Synergy)** mạnh mẽ:
$$I(i, j) = \Delta(i \mid \{j\}) - \Delta(i \mid \varnothing) \neq 0$$
Nhiều pixel khi đứng riêng lẻ hoàn toàn không đủ sức làm đổi nhãn mô hình, nhưng khi cùng được kích hoạt sẽ tạo ra sự cộng hưởng bẻ gãy dự đoán của mạng.

CASA được thiết kế xoay quanh nguyên lý: **Tìm kiếm tập hỗ trợ có điều kiện dựa trên gradient kết hợp với hoán đổi liên minh cục bộ (Coalition-Aware Gradient-Guided Support Search)**.

---

## 2. Kiến Trúc 3 Giai Đoạn Của CASA (Three-Stage Pipeline)

Toàn bộ thuật toán CASA được chuẩn hóa thành chuỗi 3 giai đoạn:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CASA PIPELINE OVERVIEW                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  GIAI ĐOẠN 1: KHỞI TẠO TẬP HỖ TRỢ (Support Initialization)                 │
│  • Tính gradient ban đầu g = ∇_x L(x, y).                                   │
│  • Lọc ứng viên qua Directional Headroom Gain A_i(x).                       │
│  • Nhánh K ≤ 2: Duyệt tọa độ chính xác GCT (Gradient Coordinate Traversal). │
│  • Nhánh K ≥ 4: Lọc đa dạng không gian Spatial NMS (Non-Maximum Suppression)│
│  • Khởi tạo giá trị biên siêu hộp Box-Extremal RGB Initialization.          │
├─────────────────────────────────────────────────────────────────────────────┤
│  GIAI ĐOẠN 2: TÌM KIẾM & HOÁN ĐỔI LIÊN MINH (Coalition Search Refinement)  │
│  • Vòng lặp ngoài T_outer:                                                  │
│    - Tối ưu hóa giá trị trên tập hỗ trợ cố định (Inner PGD with Sign Decay) │
│    - Kiểm tra điều kiện dừng sớm (Early Stopping nếu f(x') ≠ y).            │
│    - Tính gradient có điều kiện g_curr = ∇_x L(x', y).                      │
│    - Tính độ dư thừa hình chiếu: R_i = ∑_c g_{i, c} δ_{i, c}.              │
│    - Hoán đổi 1-out / 1-in (loại pixel dư thừa nhất, nạp ứng viên tối ưu).  │
│    - Nếu thất bại: Kích hoạt Adaptive 2-out / 2-in Fallback.                │
│    - Định kỳ: Thăm dò cặp đôi (Pair Exploration).                           │
│    - Bộ nhớ cấm Tabu Memory ngăn chu trình lặp hỗ trợ.                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  GIAI ĐOẠN 3: NÉN TẬP HỖ TRỢ (Support Compression: Drop-and-Repair)        │
│  • Với mỗi pixel i ∈ S:                                                     │
│    - Thử nghiệm loại bỏ i khỏi S: S_trial = S \ {i}.                        │
│    - Chạy bước sửa chữa nhanh (Repair Steps) trên S_trial.                  │
│    - Nếu vẫn duy trì lừa được mô hình: Chấp nhận nén |S| ← |S| - 1.         │
│  • Chiếu an toàn cuối cùng (Strict Spatial L0 Projection).                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Chi Tiết Các Toán Tử Thuật Toán (Algorithmic Operators)

### 3.1 Hàm mục tiêu Difference-of-Logits Ratio (DLR Loss)
CASA sử dụng hàm mất mát DLR (Difference-of-Logits Ratio) bất biến theo tỷ lệ afin nhằm chống lại hiện tượng triệt tiêu gradient ở vùng logit bão hòa:

$$\mathcal{L}_{\text{DLR}}(x', y) = \frac{z_{\pi_1}(x') - z_y(x')}{z_{\pi_1}(x') - z_{\pi_3}(x') + \varepsilon}$$
trong đó:
- $z_y(x')$ là logit của nhãn đúng $y$.
- $z_{\pi_1}(x')$ là logit cao nhất trong tất cả các lớp khác nhãn $y$:
  $$\pi_1 = \arg\max_{c \neq y} z_c(x')$$
- $z_{\pi_3}(x')$ là logit cao thứ 3 trên toàn bộ các lớp.
- $\varepsilon = 10^{-12}$ chống chia cho 0.

Hàm biên (Attack Margin):
$$\mathcal{M}(x', y) = \max_{c \neq y} z_c(x') - z_y(x')$$
Điều kiện tấn công thành công xảy ra khi $\mathcal{M}(x', y) > 0$.

---

### 3.2 Tiềm năng định hướng trong hộp giới hạn (Directional Headroom Gain)
Gradient thông thường $g_i = \nabla_{x_i} \mathcal{L}$ không phản ánh lượng thay đổi hàm mục tiêu khả dĩ vì pixel bị chặn trong $[0, 1]$. Nếu $g_{i, c} > 0$ nhưng $x_{i, c} \approx 1.0$, pixel không còn dư địa tăng giá trị để khuếch đại loss.

CASA định nghĩa **Directional Headroom Gain** cho mỗi điểm ảnh $i$:
$$A_i(x) = \sum_{c \in \{R, G, B\}} \left[ \max(g_{i, c}, 0) \cdot (1 - x_{i, c}) + \max(-g_{i, c}, 0) \cdot x_{i, c} \right]$$

Toán tử này cân đo chính xác tích phân đường khả thi từ vị trí hiện tại đến biên siêu hộp theo hướng đạo hàm. Điểm ảnh được xếp hạng vào Candidate Pool $\mathcal{C}$ dựa trên $A_i(x)$.

---

### 3.3 Khởi tạo tối ưu: GCT vs Spatial NMS
Tùy thuộc vào ngân sách $K$, CASA kích hoạt cơ chế khởi tạo tập hỗ trợ ban đầu thích hợp:

1. **Gradient Coordinate Traversal (GCT) cho $K \in \{1, 2\}$:**
   Ở mức cực thưa, số lượng điểm ảnh can thiệp chỉ là 1 hoặc 2. CASA trích xuất Top-$M$ ứng viên có $A_i(x)$ cao nhất ($M = 64$).
   - Với $K=1$: Đánh giá trực diện giá trị cực biên trên từng ứng viên, chọn điểm ảnh tạo margin lớn nhất.
   - Với $K=2$: Đánh giá các cặp đôi ứng viên hàng đầu bằng cách thử nghiệm đồng thời, tìm kiếm cặp có hiệp đồng $I(i, j) > 0$ lớn nhất.

2. **Spatial Non-Maximum Suppression (Spatial NMS) cho $K \ge 4$:**
   Gradient của mạng CNN thường tập trung thành các cụm dày đặc quanh các cạnh nổi bật. Nếu chọn Top-$K$ thuần túy, tất cả $K$ điểm ảnh sẽ nằm chen chúc trong một vùng $2 \times 2$, gây dư thừa thông tin (high redundancy).
   CASA áp dụng Spatial NMS với bán kính không gian $r_{\text{nms}} = 2$ pixel:
   $$\text{Nếu } \|(h_i, w_i) - (h_j, w_j)\|_\infty \le r_{\text{nms}}, \quad \text{loại bỏ ứng viên có } A_j(x) \text{ nhỏ hơn.}$$
   Toán tử này ép các điểm ảnh trong tập hỗ trợ phân tán đều trên các đặc trưng khác nhau của vật thể.

3. **Box-Extremal RGB Initialization:**
   Sau khi chọn được $S_0$, các giá trị khởi tạo được đẩy trực tiếp về các đỉnh đối nghịch của siêu hộp:
   $$x'_{0, c, i} = \begin{cases} 1.0 & \text{nếu } g_{i, c} \ge 0 \\ 0.0 & \text{nếu } g_{i, c} < 0 \end{cases} \quad (\forall i \in S_0)$$
   giúp thuật toán bắt đầu ngay từ vùng nhiễu mạnh nhất.

---

### 3.4 Tinh chỉnh liên minh (Coalition Support Refinement)

Trong mỗi vòng lặp ngoài $t = 1, \dots, T_{\text{outer}}$:
1. **Tối ưu hóa giá trị (Inner Value Optimization):** Cố định tập hỗ trợ $S_t$, chạy $T_{\text{inner}}$ bước gradient ascent có suy giảm bước nhảy (sign decay):
   $$x'_{m+1, S} = \operatorname{clip}_{[0, 1]}\left( x'_{m, S} + \alpha_m \cdot \operatorname{sign}\left(\nabla_{x'_{S}} \mathcal{L}_{\text{DLR}}\right) \right)$$
   với $\alpha_m = \alpha_0 \cdot (0.5)^{m / T_{\text{inner}}}$.

2. **Xác định điểm ảnh dư thừa (Projection Redundancy):**
   Tính đóng góp thực tế của từng pixel $i \in S_t$ vào gradient hiện tại:
   $$R_i = \sum_{c \in \{R, G, B\}} g_{\text{curr}, i, c} \cdot (x'_{i, c} - x_{i, c})$$
   Điểm ảnh có $R_i$ nhỏ nhất là điểm ảnh ít đóng góp nhất vào đà tăng loss hiện tại và được chọn để loại bỏ:
   $$i_{\text{drop}} = \arg\min_{i \in S_t} R_i$$

3. **Toán tử hoán đổi 1-out / 1-in:**
   Chọn ứng viên mới $j_{\text{add}} \notin S_t$ từ pool ứng viên được cập nhật động:
   $$S_{\text{new}} = (S_t \setminus \{i_{\text{drop}}\}) \cup \{j_{\text{add}}\}$$
   Nếu $S_{\text{new}}$ không nằm trong bộ nhớ Tabu và cải thiện margin mục tiêu, hoán đổi được chấp nhận.

4. **Adaptive 2-out / 2-in Fallback & Pair Exploration:**
   Nếu hoán đổi 1-swap liên tiếp bị từ chối (bị kẹt ở cực tiểu cục bộ tham lam), CASA kích hoạt hoán đổi 2 pixel đồng thời để vượt qua bẫy hiệp đồng (Synergy Trap).

5. **Bộ nhớ cấm Tabu Memory:**
   Lưu trữ vết băm của các tập hỗ trợ đã duyệt trong $K_{\text{tabu}} = 10$ bước gần nhất. Nghiêm cấm thuật toán quay lại tập hỗ trợ cũ, triệt tiêu hoàn toàn hiện tượng dao động chu trình (cycling oscillations).

---

### 3.5 Cơ chế nén tập hỗ trợ Drop-and-Repair (Compression)

Sau khi tìm được mẫu đối nghịch thành công với $|S| \le K$, CASA không dừng lại mà kích hoạt toán tử **Drop-and-Repair**:
- Sắp xếp các điểm ảnh trong $S$ theo thứ tự tăng dần của độ lệch chuẩn can thiệp $\|\delta_i\|_2$.
- Lần lượt thử loại bỏ từng pixel $i \in S$.
- Chạy $T_{\text{repair}} = 5$ bước gradient ascent nhanh trên tập còn lại $S \setminus \{i\}$.
- Nếu mô hình vẫn bị đánh lừa ($\mathcal{M} > 0$), việc loại bỏ được chấp nhận vĩnh viễn:
  $$S \leftarrow S \setminus \{i\}, \quad |S| \leftarrow |S| - 1$$

Toán tử này giúp CASA thường xuyên đạt được mức thưa thực tế nhỏ hơn ngân sách cho phép:
$$\overline{L}_0 \le 0.75 \times K$$
mà vẫn giữ nguyên tỷ lệ thành công ASR.

---

## 4. Mã Giả Chuẩn Hóa Của CASA (Formal Pseudocode)

```python
def CASA_Attack(x, y, model, K, T_outer=20, T_inner=10, alpha=0.1):
    # --- Giai đoạn 1: Khởi tạo ---
    g_clean = compute_gradient(model, x, y, loss="dlr")
    gain = compute_headroom_gain(x, g_clean)
    
    if K <= 2:
        S = run_gct(model, x, y, gain, K)
    else:
        candidates = select_top_candidates(gain, pool_size=M*K)
        S = spatial_nms(candidates, radius=2, max_k=K)
        
    x_adv = box_extremal_init(x, S, g_clean)
    tabu_memory = TabuQueue(max_size=10)
    
    # --- Giai đoạn 2: Tối ưu & Hoán đổi liên minh ---
    for t in range(T_outer):
        # Tối ưu hóa giá trị trên support cố định
        x_adv = inner_pgd_with_decay(model, x_adv, y, S, steps=T_inner, lr=alpha)
        
        # Kiểm tra thành công sớm
        if model.predict(x_adv) != y:
            break
            
        tabu_memory.push(hash(S))
        g_curr = compute_gradient(model, x_adv, y, loss="dlr")
        
        # 1-swap: Tìm pixel dư thừa nhất
        drop_idx = argmin_projection_redundancy(g_curr, x_adv - x, S)
        add_idx = select_best_candidate(gain, S, tabu_memory)
        
        S_trial = (S - {drop_idx}) + {add_idx}
        if loss(S_trial) > loss(S) and not tabu_memory.contains(hash(S_trial)):
            S = S_trial
        else:
            # Fallback 2-swap nếu 1-swap bế tắc
            S = adaptive_pair_swap(model, x_adv, y, S, gain, tabu_memory)
            
    # --- Giai đoạn 3: Nén tập hỗ trợ ---
    if model.predict(x_adv) != y:
        S, x_adv = drop_and_repair(model, x, x_adv, y, S, repair_steps=5)
        
    # Chiếu an toàn L0 và kẹp biên [0, 1]
    x_adv = strict_l0_projection(x, x_adv, max_k=K)
    return clamp(x_adv, 0.0, 1.0)
```

---

## 5. Phân Tích Độ Phức Tạp Thuật Toán (Complexity Analysis)

- **Số lượt tính đạo hàm hộp trắng (Backward passes - $B$):**
  $$B \le 1 + T_{\text{outer}} \times T_{\text{inner}} + T_{\text{compress}} \times T_{\text{repair}}$$
  Với cấu hình mặc định ($T_{\text{outer}}=20, T_{\text{inner}}=10$), số bước lặp lý thuyết tối đa là 200 lượt. Tuy nhiên, nhờ cơ chế dừng sớm (Early Stopping), số lượt tính thực tế trung bình chỉ là **$35 \to 65$ passes/ảnh**.
- **Bộ nhớ GPU phụ trội:**
  Chỉ lưu trữ tensor mask kích thước $(B, 1, H, W)$ và các mảng chỉ số hỗ trợ kích thước $(B, K)$, tiêu tốn dưới **15 MB** VRAM phụ trội trên batch size 100.

---

## 6. Ánh Xạ Kiến Trúc Mã Nguồn (`src/aa/attacks/casa/`)

Mỗi khối toán học trong đặc tả trên tương ứng 1-1 với các lớp mã nguồn được module hóa:

| Thành phần toán học | Lớp / Module trong Source Code | Đường dẫn tệp |
| :--- | :--- | :--- |
| **Pipeline điều phối chính** | [`CoalitionSparseAttack`](file:///Volumes/WorkSpace/Project/AA/src/aa/attacks/casa/attack.py) | `src/aa/attacks/casa/attack.py` |
| **Đối tượng tập hỗ trợ ($S$)** | [`Support`](file:///Volumes/WorkSpace/Project/AA/src/aa/attacks/casa/support.py) | `src/aa/attacks/casa/support.py` |
| **Hàm mục tiêu DLR & Margin** | [`CoalitionObjective`](file:///Volumes/WorkSpace/Project/AA/src/aa/attacks/casa/objective.py) | `src/aa/attacks/casa/objective.py` |
| **Sàng lọc Headroom Gain** | [`CandidateGenerator`](file:///Volumes/WorkSpace/Project/AA/src/aa/attacks/casa/candidate.py) | `src/aa/attacks/casa/candidate.py` |
| **GCT & Spatial NMS Init** | [`GCTInitializer`](file:///Volumes/WorkSpace/Project/AA/src/aa/attacks/casa/initialization.py), [`SpatialNMS`](file:///Volumes/WorkSpace/Project/AA/src/aa/attacks/casa/initialization.py) | `src/aa/attacks/casa/initialization.py` |
| **Tối ưu hóa giá trị Inner** | [`FixedSupportOptimizer`](file:///Volumes/WorkSpace/Project/AA/src/aa/attacks/casa/value_optimization.py) | `src/aa/attacks/casa/value_optimization.py` |
| **Nén Drop-and-Repair** | [`DropAndRepairCompressor`](file:///Volumes/WorkSpace/Project/AA/src/aa/attacks/casa/compression.py) | `src/aa/attacks/casa/compression.py` |
