# 07. Kế Hoạch Thực Nghiệm Vận Hành (Operational Experiment Plan)

> **Dự án:** AA (Sparse Adversarial Attack Benchmark)  
> **Phương pháp trung tâm:** CASA  
> **Trạng thái:** Operational Research Plan (V2 Specification)  
> **Mục tiêu:** Quy định toàn bộ ma trận thực nghiệm cần chạy, phân kỳ các giai đoạn thực thi (Phases P0–P5), hệ thống cổng kiểm soát Go/No-Go và chiến lược phân bổ tài nguyên GPU.

---

## 1. Ma Trận Thực Nghiệm Toàn Diện (Master Experiment Matrix)

Không gian thực nghiệm của dự án là tích Descartes giữa các chiều tham số:
$$\text{Matrix} = \text{Dataset} \times \text{Architecture} \times \text{Attack} \times K \times \text{Seed}$$

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          KHÔNG GIAN THỰC NGHIỆM TỔNG THỂ                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Tập dữ liệu (Datasets):       CIFAR-10, CIFAR-100, Tiny-ImageNet (3)      │
│ • Kiến trúc mạng (Models):      ResNet-18, WideResNet-28-10, ResNet-50 (3)  │
│ • Giải thuật tấn công (Attacks): CASA, SPGD, Sparse-RS, CornerSearch,       │
│                                 Sigma-Zero, SparseFool, GSE, PGD0 (8)       │
│ • Mức ngân sách (K values):     K ∈ {1, 2, 4, 8, 16, 32, 64} (7)            │
│ • Tập Random Seed:              {42} (chính thức) hoặc {42, 123, 999} (đa hạt)│
└─────────────────────────────────────────────────────────────────────────────┘
```

Chỉ riêng cấu hình cốt lõi trên 2 tập dữ liệu đầu tiên đã tương đương:
$$2 \times 3 \times 8 \times 7 = 336 \text{ ô thực nghiệm độc lập}$$
Do đó, việc tự động hóa kịch bản chạy và tối ưu hóa tài nguyên tính toán là điều kiện tiên quyết.

---

## 2. Chiến Lược Phễu Lọc Tài Nguyên Tính Toán (Compute Funnel Strategy)

Tuyệt đối **không chạy toàn bộ ma trận 10.000 mẫu ngay từ đầu**. Quá trình thực nghiệm tuân theo quy tắc phễu lọc 5 giai đoạn:

```text
  [Stage A: Smoke Run]       N = 20–32 mẫu     Runtime < 1 phút       Kiểm tra cú pháp & pipeline
          │
          ▼
  [Stage B: Dev Validation]  N = 1.000 mẫu     Runtime ~10–15 phút    Tinh chỉnh & khám phá
          │
          ▼
  [Stage C: Canonical Run]   N = 10.000 mẫu    Runtime ~2 giờ (T4)    Số liệu bài báo chính thức
          │
          ▼
  [Stage D: Multi-Seed]      N = 10.000 × 3    Chỉ chạy các mốc K=1,4 Kiểm định phương sai (H6)
          │
          ▼
  [Stage E: Generalization]  Dataset mới       Chỉ chạy CASA + Top-3  Xác nhận mở rộng quy mô
```

---

## 3. Lộ Trình Phân Kỳ Thực Thi (Phased Roadmap P0 – P5)

### Phase 0: Kiểm soát Tính Đúng Đắn & Nền Tảng (P0 — Correctness First)
*Mục tiêu: Đảm bảo không còn bất kỳ lỗi kỹ thuật, sai lệch hạch toán hay lỗ hổng toàn vẹn nào trước khi chạy tính toán nặng.*
- Chuẩn hóa hạch toán queries và đạo hàm ($F + 2B$).
- Khóa chặt mã băm SHA256 cho toàn bộ checkpoint mô hình.
- Rà soát tính bất biến $\text{ASR} + \text{CRA} \equiv 100\%$.
- Thiết lập validator tự động kiểm tra định dạng JSON thô.

### Phase 1: Chuẩn Hóa Dữ Liệu & Danh Mục Checkpoint (P1 — Foundation)
*Mục tiêu: Đóng băng các tập dữ liệu và checkpoint mục tiêu.*
- Hoàn thiện `DatasetRegistry` và xác thực fingerprint của tập test.
- Hoàn thiện `ModelRegistry`, xác nhận độ chính xác sạch (Clean Acc $\ge 94.8\%$ trên CIFAR-10).
- Huấn luyện và nghiệm thu các checkpoint còn thiếu cho CIFAR-100 và WRN-28-10.

### Phase 2: Thẩm Định Benchmark Tấn Công Chính (P2 — Main Attack Evidence)
*Mục tiêu: Chạy toàn diện 8 baseline và CASA trên tập kiểm thử chuẩn CIFAR-10.*
- Chạy toàn bộ 10.000 mẫu trên ResNet-18 qua 7 mốc $K$.
- Chạy cross-architecture trên WRN-28-10 và ResNet-50.
- Sinh bảng so sánh ASR@$K$, CRA@$K$, achieved $L_0$, PSNR, SSIM và runtime.

### Phase 3: Nghiên Cứu Chuyên Sâu & Phân Rã Thành Phần (P3 — Scientific Analysis)
*Mục tiêu: Giải thích bản chất toán học "Vì sao CASA thành công?".*
- Chạy phân rã 10 biến thể thuật toán (Ablation study trên CIFAR-10/ResNet-18).
- Đo lường trực tiếp mức độ tương tác liên minh $I(i, j) \neq 0$ và tỷ lệ giải cứu của Pair-Swap.
- Khảo sát độ nhạy siêu tham số (bước lặp $T$, learning rate $\alpha$, bán kính NMS).
- Phân tích thất bại (Failure Analysis) và counterfactual cases.

### Phase 4: Mở Rộng Quy Mô & Tổng Quát Hóa (P4 — Generalization)
*Mục tiêu: Chứng minh tính phổ quát của giải thuật.*
- Thực thi benchmark trên CIFAR-100 (100 lớp).
- Thực thi benchmark trên Tiny-ImageNet ($64 \times 64$, $4.096$ pixel).

### Phase 5: Nhánh Nghiên Cứu Phòng Thủ (P5 — Defense Research Branch)
*Mục tiêu: Khảo sát độ bền vững thực tế trước tấn công thưa.*
- Đánh giá các bộ lọc tiền xử lý (Median, Blur, JPEG, TVM) ở chế độ Vô thức (Oblivious).
- Đánh giá thích ứng với BPDA để đo lường khoảng cách bẻ gãy (Adaptive Gap).
- Khảo sát sơ bộ mô hình huấn luyện đối nghịch thưa CASA-AT.

---

## 4. Hệ Thống Cổng Kiểm Soát Nghiên Cứu (Go / No-Go Decision Gates)

Mỗi giai đoạn chỉ được chuyển tiếp khi vượt qua cổng nghiệm thu tương ứng:

| Cổng | Tên cổng | Tiêu chí vượt qua (PASS Criteria) | Quyết định nếu FAIL |
| :---: | :--- | :--- | :--- |
| **Gate 0** | **Method Freeze** | Giải thuật CASA ổn định, không sửa thêm heuristic rời rạc; mô hình toán học khớp 100% với code. | Không chạy benchmark lớn. |
| **Gate 1** | **Benchmark Correctness** | L0 luôn $\le K$, ảnh sau nhiễu nằm trong $[0, 1]$, hạch toán $F + 2B$ chính xác, không dùng `\|\| true`. | Sửa pipeline đo lường. |
| **Gate 2** | **Single-Setting SOTA** | CASA vượt trội baseline trên CIFAR-10/ResNet-18 ở vùng $K \le 4$ với $p < 0.01$ (McNemar test). | Tinh chỉnh toán tử tìm kiếm. |
| **Gate 3** | **Multi-Model Stability** | Xu hướng cải thiện ASR duy trì nhất quán trên WRN-28-10 và ResNet-50. | Kiểm tra hiện tượng overfit kiến trúc. |
| **Gate 4** | **Multi-Dataset Validity** | CASA duy trì hiệu quả cao hơn các baseline trên CIFAR-100. | Tái thẩm định hàm mất mát DLR. |
| **Gate 5** | **Scientific Evidence** | Hoàn thành Ablation 10 biến thể, chứng minh vai trò độc lập của GCT, NMS và 1-swap. | Không công bố paper. |
| **Gate 6** | **Reproducibility** | Clean-clone script chạy tự động 1 lệnh tái lập chính xác bảng số liệu từ JSON thô. | Sửa runner và manifest. |
| **Gate 7** | **Publication Ready** | Hoàn tất manuscript, figures/tables sinh tự động, không có claim vượt quá bằng chứng. | Bài báo sẵn sàng nộp. |
