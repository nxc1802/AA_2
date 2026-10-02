# 01. Tổng Quan Nghiên Cứu & Xác Lập Phạm Vi (Research Overview & Scope)

> **Dự án:** AA (Sparse Adversarial Attack Benchmark)  
> **Phương pháp trung tâm:** CASA (*Coalition-Aware Sparse Adversarial Attack*)  
> **Trạng thái:** Canonical Documentation (V2 Specification)  
> **Mục tiêu:** Bản đồ nghiên cứu tổng thể, định nghĩa bài toán, câu hỏi nghiên cứu (RQs), đóng góp cốt lõi và phân định phạm vi khoa học.

---

## 1. Bối Cảnh & Bài Toán Nghiên Cứu (Problem Formulation)

Trong lĩnh vực an toàn học máy (Machine Learning Security), các đòn tấn công đối nghịch thưa thớt (*Sparse Adversarial Attacks*) dưới chuẩn $L_0$ đóng vai trò nền tảng để kiểm tra độ bền vững nội tại của mạng nơ-ron sâu. Khác với các đòn tấn công dày (*dense attacks* như FGSM, PGD $L_\infty$ hoặc $L_2$) tác động lên toàn bộ hàng nghìn pixel của ảnh, tấn công thưa thớt giới hạn nghiêm ngặt số lượng điểm ảnh bị can thiệp:

$$\|\delta\|_{0, \text{spatial}} \le K$$

với ngân sách $K$ thường rất nhỏ ($K \in \{1, 2, 4, 8, 16, 32, 64\}$) trên không gian ảnh $H \times W$.

### Thách thức cốt tử: Tối ưu hóa tổ hợp tập hỗ trợ (Combinatorial Support Selection)
Tấn công thưa thớt không gian thực chất là một bài toán tối ưu hóa tổ hợp phức tạp:
1. **Tìm kiếm tập hỗ trợ (Support Selection):** Chọn ra một tập con $S \subseteq \{1, \dots, H \times W\}$ với $|S| \le K$ tọa độ không gian hiệu quả nhất.
2. **Tối ưu hóa giá trị liên tục (Continuous Value Optimization):** Tối ưu hóa biên độ nhiễu liên tục $\delta_S \in [0, 1]^3$ trên các kênh màu RGB tại các tọa độ thuộc $S$.

Hai bài toán này phụ thuộc chéo chặt chẽ (*coupled*). Các phương pháp tấn công truyền thống thường tiếp cận theo hướng đơn giản hóa:
- **Xếp hạng điểm ảnh độc lập (Independent Saliency Ranking):** Tính gradient trên ảnh sạch hoặc saliency map rồi chọn Top-$K$ điểm ảnh có độ lớn lớn nhất (ngầm giả định lợi ích điểm ảnh là độc lập $\Delta(i \mid S) \approx \Delta(i \mid \varnothing)$). Giả định này sụp đổ ở vùng cực thưa ($K \le 4$), nơi tương tác phi tuyến tính giữa các điểm ảnh quyết định sự thành bại.
- **Nới lỏng liên tục (Continuous Relaxation):** Sử dụng các chuẩn xấp xỉ ($L_1$, tanh, homotopy) nhưng dễ rơi vào cực tiểu cục bộ và đòi hỏi chiếu gượng ép về $L_0$.
- **Tìm kiếm hộp đen thuần túy (Black-box Random Search):** Tiêu tốn hàng chục nghìn truy vấn cho mỗi ảnh (như CornerSearch, Sparse-RS), không tận dụng được thông tin đạo hàm trong kịch bản hộp trắng.

---

## 2. Hệ Thống Câu Hỏi Nghiên Cứu (Research Questions)

Dự án được định hướng bởi 5 câu hỏi nghiên cứu cốt lõi:

* **RQ1 (Attack Effectiveness & Budget Adherence):**  
  *CASA có tìm được mẫu đối nghịch thành công dưới ngân sách spatial $L_0 \le K$ hiệu quả hơn các phương pháp tấn công thưa thớt hiện có (cả hộp trắng và hộp đen) hay không? Mức độ tuân thủ ngân sách có được đảm bảo tuyệt đối không?*

* **RQ2 (Model & Dataset Generalization):**  
  *Ưu thế của CASA có duy trì ổn định khi mở rộng quy mô từ CIFAR-10 sang các bài toán phân loại phức tạp hơn (CIFAR-100) và không gian tìm kiếm lớn hơn (Tiny-ImageNet 64×64), cũng như trên nhiều họ kiến trúc mạng khác nhau (ResNet-18, WRN-28-10, ResNet-50) hay không?*

* **RQ3 (Coalition-Aware Mechanism & Synergy):**  
  *Cơ chế tìm kiếm tập hỗ trợ có điều kiện và hoán đổi liên minh (1-swap, 2-swap, pair exploration) đóng góp định lượng như thế nào vào việc thoát khỏi bẫy cực tiểu tham lam? Tương tác hiệp đồng giữa các pixel $I(i, j) \neq 0$ có thực sự đo lường được trên các mẫu khó không?*

* **RQ4 (Efficiency & Query-Computation Trade-off):**  
  *Chi phí tính toán thực tế của CASA (đo bằng wall-clock time, số lượt lan truyền xuôi $F$, lan truyền ngược $B$, và FLOP-equivalent $F + 2B$) có đạt được đường biên Pareto vượt trội so với các đối thủ mạnh hay không?*

* **RQ5 (Sparse Robustness & Adaptive Defense):**  
  *Các cơ chế phòng thủ hiện tại (lọc tiền xử lý, huấn luyện đối nghịch thưa) có tạo ra độ bền vững thực sự trước tấn công thưa thớt hay chỉ tạo hiện tượng mặt nạ gradient (gradient masking)? Huấn luyện đối nghịch dựa trên CASA (CASA-AT) có cải thiện được độ bền vững thích ứng hay không?*

---

## 3. Các Đóng Góp Khoa Học Cốt Lõi (Core Contributions)

1. **Thuật toán CASA (Coalition-Aware Sparse Adversarial Attack):**  
   Xây dựng một giải thuật tấn công thưa thớt hộp trắng thống nhất, kết hợp giữa sàng lọc ứng viên định hướng (Directional Headroom Gain), khởi tạo tối ưu (GCT cho $K \le 2$, Spatial NMS cho $K \ge 4$), tối ưu hóa liên tục trên tập hỗ trợ cố định và tinh chỉnh liên minh động (Dynamic Support Refinement qua 1-swap và adaptive 2-swap).

2. **Khung lý thuyết Tối ưu hóa Tập hỗ trợ có điều kiện:**  
   Mô hình hóa bài toán tấn công thưa thớt dưới lăng kính lý thuyết trò chơi liên minh, chỉ ra giới hạn của việc xếp hạng điểm ảnh độc lập và chứng minh bằng thực nghiệm sự tồn tại của tương tác hiệp đồng $I(i, j) \neq 0$.

3. **Cơ chế nén tập hỗ trợ Drop-and-Repair:**  
   Thuật toán nén tập hỗ trợ hậu tối ưu giúp giảm $L_0$ thực tế xuống thấp hơn đáng kể so với ngân sách trần $K$ mà vẫn bảo toàn tỷ lệ tấn công thành công (ASR).

4. **Khung kiểm chuẩn (Benchmark Suite) nghiêm ngặt & chuẩn hóa:**  
   Thiết lập Single Source of Truth (SSOT) cho giao thức kiểm thử $L_0$, chuẩn hóa mẫu số tính ASR (chỉ tính trên các mẫu phân loại đúng ban đầu), kiểm soát tính toàn vẹn checkpoint qua SHA256, và đo lường chi phí tính toán công bằng ($F + 2B$).

5. **Đánh giá phòng thủ thích ứng (Adaptive Defense Evaluation):**  
   Triển khai bộ chuyển đổi BPDA (*Backward Pass Differentiable Approximation*) và EOT (*Expectation Over Transformations*) để kiểm tra toàn diện hiện tượng mặt nạ gradient trên các kỹ thuật lọc ảnh.

---

## 4. Phân Định Phạm Vi Nghiên Cứu (Research Scope & Tracks)

Để đảm bảo tính khả thi và tập trung nguồn lực tính toán, dự án phân tách rõ hai hướng nghiên cứu:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       RESEARCH SCOPE & TRACKS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  TRACK A: ATTACK-FIRST (Trọng tâm bài báo chính - Manuscript Primary)      │
│  • Khóa chặt giải thuật CASA và toán tử tìm kiếm liên minh.                 │
│  • Đánh giá trên 8 baseline đối thủ (SPGD, Sparse-RS, Sigma-Zero, ...)      │
│  • Đa kiến trúc (ResNet-18, WRN-28-10, ResNet-50) & đa dataset.            │
│  • Phân rã thành phần (Ablation), độ nhạy tham số và phân tích thất bại.    │
├─────────────────────────────────────────────────────────────────────────────┤
│  TRACK B: DEFENSE RESEARCH BRANCH (Nghiên cứu độc lập / Mở rộng)            │
│  • Khảo sát các bộ lọc tiền xử lý (Median, Blur, JPEG, TVM).               │
│  • Đánh giá thích ứng với BPDA/EOT để bóc trần gradient masking.           │
│  • Huấn luyện đối nghịch thưa CASA-AT (đang ở trạng thái phát triển/kiểm nghiệm).│
└─────────────────────────────────────────────────────────────────────────────┘
```

### Bảng phân định trạng thái các cấu phần:

| Hạng mục | Trạng thái | Ghi chú khoa học |
| :--- | :---: | :--- |
| **CASA Attack Algorithm** | **Implemented & Frozen** | Khóa cấu trúc code tại `src/aa/attacks/casa/`. |
| **8 Baseline Adapters** | **Verified** | Đồng bộ pinned commit và upstream provenance trong `THIRD_PARTY.md`. |
| **Benchmark Protocol & SSOT** | **Frozen** | Khóa chặt tại [`02_threat_model_and_protocol.md`](file:///Volumes/WorkSpace/Project/AA/docs/02_threat_model_and_protocol.md). |
| **CIFAR-10 10k Benchmark** | **Evaluated** | Đầy đủ kết quả 7 mốc $K$, artifacts JSON và kiểm định Wilson CI. |
| **Ablation Study (10 biến thể)** | **Evaluated** | Xác thực vai trò của từng module (GCT, NMS, 1-swap, Drop-repair). |
| **CIFAR-100 & Tiny-ImageNet** | **Evaluated / In Progress** | Kiểm thử mở rộng quy mô bài toán và độ phân giải. |
| **Defense Preprocessing + BPDA** | **Implemented** | Bộ lọc và adapter vi phân xấp xỉ hoàn chỉnh trong `src/aa/defenses/`. |
| **CASA-AT Adversarial Training** | **Planned / Exploratory** | Nghiên cứu độc lập, không đưa claim sớm vào bài báo tấn công. |

---

## 5. Phân Tầng Tuyên Bố Khoa Học (Publication Claim Hierarchy)

Mọi tuyên bố trong bài báo khoa học phải tương xứng với bằng chứng thực nghiệm đã thẩm định:

```text
Claim Level 1: CASA vượt trội trên CIFAR-10 với backbone ResNet-18.
               ├── [ĐÃ ĐẠT: Thẩm định trên 10.000 mẫu test split chuẩn]
               ▼
Claim Level 2: CASA cải thiện hiệu quả tấn công nhất quán trên nhiều kiến trúc CNN.
               ├── [ĐÃ ĐẠT: ResNet-18, WRN-28-10, ResNet-50]
               ▼
Claim Level 3: CASA tổng quát hóa qua nhiều bài toán và độ phân giải.
               ├── [ĐẠT MỘT PHẦN: CIFAR-10, CIFAR-100; đang hoàn thiện Tiny-ImageNet]
               ▼
Claim Level 4: CASA thiết lập đường biên Pareto vượt trội về hiệu năng/chi phí ($F + 2B$).
               ├── [ĐÃ ĐẠT: So sánh trực tiếp với SPGD, Sparse-RS và CornerSearch]
               ▼
Claim Level 5: Tuyên bố SOTA toàn diện (Universal State-of-the-Art).
               └── [CHỈ TUYÊN BỐ KHI: Toàn bộ baselines hoàn tất dưới cùng giao thức chuẩn]
```

---

## 6. Sơ Đồ Điều Hướng Tài Liệu (Documentation Roadmap)

Hệ thống tài liệu chuẩn của dự án gồm 14 tài liệu canonical trong thư mục `docs/`:

1. [`01_research_overview.md`](file:///Volumes/WorkSpace/Project/AA/docs/01_research_overview.md) — Tổng quan, câu hỏi nghiên cứu, phạm vi & đóng góp.
2. [`02_threat_model_and_protocol.md`](file:///Volumes/WorkSpace/Project/AA/docs/02_threat_model_and_protocol.md) — Threat model, định nghĩa metric & giao thức SSOT.
3. [`03_casa_method.md`](file:///Volumes/WorkSpace/Project/AA/docs/03_casa_method.md) — Đặc tả giải thuật CASA, toán tử & ánh xạ code.
4. [`04_datasets.md`](file:///Volumes/WorkSpace/Project/AA/docs/04_datasets.md) — Ma trận dữ liệu, tiền xử lý & hash kiểm tra.
5. [`05_models_and_training.md`](file:///Volumes/WorkSpace/Project/AA/docs/05_models_and_training.md) — Mô hình mục tiêu, quy trình huấn luyện & cổng nghiệm thu checkpoint.
6. [`06_baselines.md`](file:///Volumes/WorkSpace/Project/AA/docs/06_baselines.md) — Phân loại đối thủ, cấu hình adapter & tiêu chí so sánh.
7. [`07_experiment_plan.md`](file:///Volumes/WorkSpace/Project/AA/docs/07_experiment_plan.md) — Kế hoạch thực nghiệm vận hành, phân kỳ P0–P5 & cổng Go/No-Go.
8. [`08_attack_benchmark.md`](file:///Volumes/WorkSpace/Project/AA/docs/08_attack_benchmark.md) — Kết quả thực nghiệm tấn công, bảng so sánh & phân tích Pareto.
9. [`09_ablation_and_analysis.md`](file:///Volumes/WorkSpace/Project/AA/docs/09_ablation_and_analysis.md) — Phân rã thành phần, tương tác cặp đôi & phân tích thất bại.
10. [`10_defense_research.md`](file:///Volumes/WorkSpace/Project/AA/docs/10_defense_research.md) — Nhánh phòng thủ, tiền xử lý, kiểm định BPDA/EOT & mặt nạ gradient.
11. [`11_reproducibility.md`](file:///Volumes/WorkSpace/Project/AA/docs/11_reproducibility.md) — Sổ tay tái lập, hướng dẫn Kaggle GPU & xác thực môi trường.
12. [`12_artifacts_and_results.md`](file:///Volumes/WorkSpace/Project/AA/docs/12_artifacts_and_results.md) — Bản đồ kết quả, cấu trúc JSON thô & mã băm SHA256.
13. [`13_limitations.md`](file:///Volumes/WorkSpace/Project/AA/docs/13_limitations.md) — Giới hạn thuật toán, biên thực nghiệm & các tuyên bố không hỗ trợ.
14. [`14_publication_checklist.md`](file:///Volumes/WorkSpace/Project/AA/docs/14_publication_checklist.md) — Cổng kiểm soát xuất bản, tiêu chuẩn DoD & trạng thái nghiệm thu.
