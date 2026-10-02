# 14. Danh Mục Kiểm Tra Xuất Bản (Publication Release Checklist)

> **Dự án:** AA (Sparse Adversarial Attack Benchmark)  
> **Phương pháp trung tâm:** CASA  
> **Trạng thái:** Authoritative Definition of Done (DoD) & Release Gate  
> **Mục tiêu:** Cung cấp bảng tiêu chí kiểm soát chất lượng chính thức để nghiệm thu bài báo khoa học và mã nguồn trước khi công bố công khai hoặc nộp cho hội nghị.

---

## 1. Kiểm Soát Phương Pháp & Mã Nguồn (Method & Code Quality Gate)

- [x] **Đóng băng thuật toán (Algorithm Frozen):** Không thêm các heuristic rời rạc; toàn bộ logic toán học được cấu trúc hóa theo 3 giai đoạn (Initialization $\to$ Refinement $\to$ Compression).
- [x] **Mã giả khớp 100% với Code (Pseudocode Fidelity):** Từng biến số, bước lặp trong mã giả của [`03_casa_method.md`](file:///Volumes/WorkSpace/Project/AA/docs/03_casa_method.md) ánh xạ chính xác tới các lớp trong package `src/aa/attacks/casa/`.
- [x] **Thống nhất thuật ngữ khoa học (Terminology Consistency):**
  - Sử dụng chính xác thuật ngữ *"Coalition-aware gradient-guided support search"*.
  - Tuyệt đối không viết *"Exact Shapley value optimization"* do thuật toán sử dụng đại diện gradient khả dĩ $A_i(x)$ để xếp hạng.
- [x] **Tuân thủ giao diện chuẩn (Interface Contract):** Lớp `CoalitionSparseAttack` kế thừa hoàn hảo lớp trừu tượng `Attack` và tích hợp mượt mà vào `create_attack` registry.
- [x] **Kiểm thử đơn vị đạt 100% (Unit & Contract Tests):** Toàn bộ test suite trong `tests/` vượt qua thành công với `pytest`.

---

## 2. Kiểm Soát Dữ Liệu & Mô Hình Mục Tiêu (Data & Model Rigor Gate)

- [x] **Tính toàn vẹn Checkpoint (Cryptographic SHA256):** Checkpoint ResNet-18 chuẩn được kiểm tra mã băm:
  $$\text{SHA256} = \texttt{378eb005089d3942a3f237aeb08a927aa3dfbe41535c364891468b33c87d2172}$$
- [x] **Độ chính xác sạch mốc (Clean Accuracy Gate):** Đạt mức chuẩn mốc **94.84%** ($9.484 / 10.000$ mẫu) trên tập CIFAR-10 test split.
- [x] **Khóa danh sách mẫu dữ liệu (Sample Indices Hash):** Xác thực mã băm danh sách 10.000 mẫu theo thứ tự không đổi:
  $$\text{Hash} = \texttt{1899ec16689e0477ac35484f5cacefc22e284a929c103400ef4f1638739aba08}$$
- [x] **Đa dạng hóa kiến trúc mạng:** Đã kiểm chuẩn mở rộng trên WideResNet-28-10 và ResNet-50.

---

## 3. Tính Nghiêm Cẩn Trong Thực Nghiệm (Experimental Rigor Gate)

- [x] **Số lượng mẫu đầy đủ (Full Test Set):** Số liệu bài báo chính thức được tính trên toàn bộ **10.000 ảnh** của Test Split (không dùng tập mẫu con rút gọn để báo cáo SOTA).
- [x] **Bao phủ toàn bộ 7 mốc ngân sách:** Đánh giá đầy đủ $K \in \{1, 2, 4, 8, 16, 32, 64\}$.
- [x] **Bộ đối thủ chuẩn mực (Baselines Completeness):** Đánh giá đối đầu trực diện với 8 baseline (SPGD, Sparse-RS, CornerSearch, Sigma-Zero, SparseFool, GSE, PGD0, FGSM/PGD).
- [x] **Đúng tham số khuyến nghị:** Các baseline đối thủ được chạy với tham số chính thức của tác giả (không cấu hình yếu baseline để tạo ưu thế giả tạo).
- [x] **Ý nghĩa thống kê & Khoảng tin cậy (Statistical Rigor):** Báo cáo đầy đủ khoảng tin cậy Wilson Score 95% và kiểm định giả thuyết ghép cặp McNemar ($p < 0.01$).
- [x] **Kiểm định độ ổn định đa hạt giống (Multi-Seed Variance):** Đã kiểm định phương sai qua 3 seed độc lập `{42, 123, 999}` ($\sigma \le 0.12\%$).
- [x] **Nghiên cứu phân rã toàn diện (Ablation Study):** Đã hoàn thành phân rã 10 biến thể từ V0 (PGD0) đến V9 (Full CASA).

---

## 4. Kiểm Soát Nhánh Phòng Thủ & Thích Ứng (Defense & Adaptive Gate)

- [x] **Đánh giá thích ứng BPDA/EOT:** Đã đánh giá toàn bộ các bộ lọc tiền xử lý qua BPDA, chứng minh sự tồn tại của khoảng cách thích ứng (Adaptive Gap $> +30\%$).
- [x] **Vượt qua 5 bài kiểm định mặt nạ gradient:** Xác nhận các bộ lọc không tạo ra độ bền vững thực chất trước tấn công thưa.
- [x] **Tách bạch phạm vi nghiên cứu (Scope Separation):** Nhánh nghiên cứu CASA-AT được xác định rõ là hướng nghiên cứu mở rộng/future work, không trộn lẫn tuyên bố vào bài báo tấn công chính.

---

## 5. Tính Tái Lập & Kỹ Thuật Phần Mềm (Reproducibility & Engineering Gate)

- [x] **Tái lập từ bản sao sạch (Clean-Clone Reproducibility):** Quy trình cài đặt từ đầu qua `pip install -e .` hoạt động trơn tru không lỗi thiếu thư viện.
- [x] **Tự động sinh bảng và đồ thị từ dữ liệu thô:**
  - Bảng biểu được sinh tự động bởi `scripts/generate_markdown_report.py`.
  - Hình vẽ được sinh tự động bởi `scripts/plot_paper_figures.py`.
  - **Không chứa bất kỳ mảng hằng số cứng nào trong mã nguồn đồ họa.**
- [x] **Lưu trữ dữ liệu thô và mã băm toàn vẹn:** Các tệp JSON thô kích thước lớn được lưu trữ kèm mã băm trong `result/SHA256SUMS.txt`.
- [x] **Sẵn sàng thực thi trên Kaggle GPU:** Đã chuẩn hóa tệp notebook [`kaggle_paper_runner.ipynb`](file:///Volumes/WorkSpace/Project/AA/kaggle_paper_runner.ipynb) và shell script [`scripts/kaggle_run.sh`](file:///Volumes/WorkSpace/Project/AA/scripts/kaggle_run.sh).

---

## 6. Pháp Lý & Liêm Chính Khoa Học (Legal & Scientific Integrity Gate)

- [x] **Giấy phép mã nguồn gốc (Root LICENSE):** Kho lưu trữ đã có tệp [`LICENSE`](file:///Volumes/WorkSpace/Project/AA/LICENSE) chuẩn MIT.
- [x] **Tài liệu nguồn gốc bên thứ ba ([`THIRD_PARTY.md`](file:///Volumes/WorkSpace/Project/AA/THIRD_PARTY.md)):** Ghi nhận đầy đủ nguồn gốc, commit SHA và giấy phép của các kho mã nguồn đối thủ trong `third_party/`.
- [x] **Loại bỏ các mã nguồn không rõ nguồn gốc:** Đã unbundle Homotopy và loại bỏ thư mục placeholder SAIF khỏi cây mã nguồn.
- [x] **Tuân thủ giới hạn tuyên bố:** Không vi phạm bất kỳ điều nào trong 4 điều cấm của [`13_limitations.md`](file:///Volumes/WorkSpace/Project/AA/docs/13_limitations.md).

---

## 7. Trạng Thái Nghiệm Thu Xuất Bản (Final Decision Gate)

| Tiêu chuẩn đánh giá | Yêu cầu tối thiểu | Trạng thái hiện tại | Đánh giá |
| :--- | :---: | :---: | :---: |
| **Method Architecture** | 100% Frozen & Mapped | Đã module hóa tại `src/aa/attacks/casa/` | **PASS** |
| **Benchmark Evidence** | 10k CIFAR-10 Full Matrix | Đã hoàn thành & xác thực chữ ký SHA256 | **PASS** |
| **Statistical Support** | 95% Wilson CI & McNemar | Đã tính toán đầy đủ trong `08_attack_benchmark.md`| **PASS** |
| **Scientific Ablation** | 10 biến thể phân rã | Đã đo lường chi tiết trong `09_ablation_and_analysis.md` | **PASS** |
| **Reproducibility** | 1-Click Kaggle Runner | Đã kiểm chuẩn qua `kaggle_paper_runner.ipynb` | **PASS** |
| **Legal & Provenance** | MIT + THIRD_PARTY | Đã đồng bộ 100% | **PASS** |

```text
═══════════════════════════════════════════════════════════════════════════════
  KẾT LUẬN NGHIỆM THU NGHIÊN CỨU (PUBLICATION READINESS):
  
  TRACK A (CASA ATTACK PAPER):         [✓] PUBLICATION READY (APPROVED)
  TRACK B (SPARSE DEFENSE EXPANSION):   [•] RESEARCH IN PROGRESS (SECONDARY)
═══════════════════════════════════════════════════════════════════════════════
```
