# 12. Danh Mục Kết Quả & Lưu Trữ Dữ Liệu Thô (Artifacts & Results Index)

> **Dự án:** AA (Sparse Adversarial Attack Benchmark)  
> **Phương pháp trung tâm:** CASA  
> **Trạng thái:** Authoritative Provenance & Artifact Mapping  
> **Mục tiêu:** Thiết lập bản đồ truy xuất nguồn gốc từ từng bảng biểu, đồ thị trong bài báo khoa học về các tệp dữ liệu thô (JSON/Parquet), cung cấp lược đồ dữ liệu máy đọc và danh mục chữ ký điện tử SHA256.

---

## 1. Nguyên Tắc Chuỗi Dữ Liệu Bất Biến (Data Provenance Principle)

Mọi con số xuất hiện trong bài báo khoa học bắt buộc phải tuân theo chuỗi sinh tự động không có sự can thiệp thủ công:

$$\boxed{\text{Per-Sample Record (JSON/Parquet)}} \longrightarrow \boxed{\text{Summary Manifest}} \longrightarrow \boxed{\text{Markdown / LaTeX Table}} \longrightarrow \boxed{\text{Manuscript}}$$

Tuyệt đối không lưu trữ các hằng số cứng dạng `CASA = [23.37, 38.92, ...]` trong các script vẽ đồ thị.

---

## 2. Bản Đồ Truy Xuất Nguồn Gốc Bài Báo (Paper Artifact Mapping)

### 2.1 Ánh xạ các Bảng Số Liệu (Tables Mapping)

| Bảng trong bài báo | Nội dung | Tệp dữ liệu thô nguồn | Script sinh tự động | Cấu hình thực nghiệm |
| :--- | :--- | :--- | :--- | :--- |
| **Bảng 1 (Table 1)** | Mốc tham chiếu Dense ($L_\infty$) | `result/benchmark_results.json` | `scripts/generate_markdown_report.py` | `configs/paper_cifar10.yaml` |
| **Bảng 2 (Table 2)** | So sánh ASR@K đối đầu 8 baselines | `result/benchmark_results.json`<br>`result/casa_10000_results.json` | `scripts/generate_markdown_report.py` | `configs/paper_cifar10.yaml` |
| **Bảng 3 (Table 3)** | Độ chính xác bền vững CRA@K | `result/benchmark_results.json` | `scripts/generate_markdown_report.py` | `configs/paper_cifar10.yaml` |
| **Bảng 4 (Table 4)** | Mức độ nén thưa thực tế $\overline{L}_0$ | `result/casa_10000_results.json` | `scripts/plot_paper_figures.py` | `configs/paper_cifar10.yaml` |
| **Bảng 5 (Table 5)** | Phân rã 10 biến thể Ablation | `result/ablation_results.json` | `scripts/run_ablation.py` | `configs/ablation.yaml` |
| **Bảng 6 (Table 6)** | Đánh giá thích ứng BPDA phòng thủ | `result/defense_benchmark_results.json` | `scripts/defense_benchmark.py` | `configs/development.yaml` |

---

### 2.2 Ánh xạ các Hình Vẽ Đồ Thị (Figures Mapping)

| Hình trong bài báo | Mô tả hình vẽ | Đường dẫn tệp đồ thị | Tệp dữ liệu nguồn | Script đồ họa |
| :--- | :--- | :--- | :--- | :--- |
| **Hình 1 (Figure 1)**| Sơ đồ thuật toán CASA 3 giai đoạn | Vector Diagram | [`03_casa_method.md`](file:///Volumes/WorkSpace/Project/AA/docs/03_casa_method.md) | Mermaid / Vector tool |
| **Hình 2 (Figure 2)**| Đường cong ASR@K so với baselines | [`docs/assets/figure2_asr_vs_k.png`](file:///Volumes/WorkSpace/Project/AA/docs/assets/figure2_asr_vs_k.png) | `result/benchmark_results.json` | `scripts/plot_paper_figures.py` |
| **Hình 3 (Figure 3)**| Ngân sách trần $K$ vs Actual $L_0$ | [`docs/assets/figure3_actual_l0_vs_k.png`](file:///Volumes/WorkSpace/Project/AA/docs/assets/figure3_actual_l0_vs_k.png) | `result/casa_10000_results.json` | `scripts/plot_paper_figures.py` |
| **Hình 4 (Figure 4)**| Đường biên hiệu quả Pareto (FLOPs) | [`docs/assets/figure4_efficiency_frontier.png`](file:///Volumes/WorkSpace/Project/AA/docs/assets/figure4_efficiency_frontier.png) | `result/benchmark_results.json` | `scripts/plot_paper_figures.py` |
| **Hình 5 (Figure 5)**| Tích lũy cải thiện Ablation | [`docs/assets/figure5_ablation_progression.png`](file:///Volumes/WorkSpace/Project/AA/docs/assets/figure5_ablation_progression.png) | `result/ablation_results.json` | `scripts/plot_paper_figures.py` |
| **Hình 6 (Figure 6)**| Bản đồ nhiệt không gian Spatial NMS | [`docs/assets/figure6_spatial_heatmap.png`](file:///Volumes/WorkSpace/Project/AA/docs/assets/figure6_spatial_heatmap.png) | `result/casa_10000_results.json` | `scripts/plot_paper_figures.py` |
| **Hình 7 (Figure 7)**| Đồ thị độ nhạy siêu tham số | [`docs/assets/figure7_hyperparam_sensitivity.png`](file:///Volumes/WorkSpace/Project/AA/docs/assets/figure7_hyperparam_sensitivity.png) | `result/failure_analysis.json` | `scripts/plot_paper_figures.py` |

---

## 3. Danh Mục Chữ Ký Điện Tử SHA256 (Checksum Registry)

Các tệp kết quả thô chính thức được lưu trữ kèm mã băm SHA256 bất biến (tại `result/SHA256SUMS.txt`):

```text
37509320fd83901bac8702e7ba2b1eb8d35679dd1e03ccf07ce96d58debe5628  ablation_results.json
a38723f3772aa92d56413ea8782b84b7ed6823fe827513941d2d0ddc92ad9792  baselines_spgd_sigmazero_10k.json
59a0bf5cb442ef0d29256842007df178c14b2d350a493e7593e6dd90cfa66bbf  casa_10000_results.json
c8046a40f933300d157ccdf075507984b991eb82524cd560b0e91a00dc6f23aa  cifar10_10k_artifacts.tar.gz
2657211c15ba207adad6ab3e21db7c6a66c67aa6a170fabf0b272add2e10c234  failure_analysis.json
```

Lệnh xác thực tính toàn vẹn:
```bash
cd result && shasum -a 256 -c SHA256SUMS.txt
```

---

## 4. Đặc Tả Lược Đồ Dữ Liệu Máy Đọc (JSON Schemas)

### 4.1 Lược đồ tệp tổng hợp `summary.json`
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "BenchmarkRunSummary",
  "type": "object",
  "required": ["experiment_id", "dataset", "model", "attack", "budget_k", "metrics", "provenance"],
  "properties": {
    "experiment_id": {"type": "string"},
    "dataset": {"type": "string"},
    "model": {"type": "string"},
    "attack": {"type": "string"},
    "budget_k": {"type": "integer"},
    "metrics": {
      "type": "object",
      "required": ["clean_accuracy", "asr", "cra", "achieved_l0", "l2_distortion", "psnr_db", "ssim"],
      "properties": {
        "clean_accuracy": {"type": "number"},
        "asr": {"type": "number"},
        "cra": {"type": "number"},
        "achieved_l0": {"type": "number"},
        "l2_distortion": {"type": "number"},
        "psnr_db": {"type": "number"},
        "ssim": {"type": "number"}
      }
    },
    "provenance": {
      "type": "object",
      "required": ["git_commit", "checkpoint_sha256", "seed", "timestamp"]
    }
  }
}
```

### 4.2 Lược đồ bản ghi từng mẫu `per_sample` (JSON Lines / Parquet)
Mỗi mẫu ảnh được lưu trữ thành một bản ghi độc lập phục vụ phân tích phương sai và kiểm định paired McNemar:
```json
{
  "sample_index": 421,
  "clean_label": 3,
  "clean_pred": 3,
  "clean_correct": true,
  "adv_pred": 8,
  "success": true,
  "budget_k": 4,
  "achieved_l0": 3,
  "l2": 0.384,
  "linf": 0.992,
  "psnr": 29.8,
  "ssim": 0.951,
  "forward_evals": 38,
  "backward_evals": 35,
  "runtime_ms": 132.4,
  "final_support": [142, 178, 512]
}
```

---

## 5. Lưu Trữ Dữ Liệu Ngoại Vi (External Hosting Plan)

Do kích thước các tệp kết quả thô per-sample và model weights lớn hơn giới hạn khuyến nghị của Git repo:
- **Git Repository:** Chỉ lưu trữ mã nguồn, tệp cấu hình YAML, tài liệu markdown, đồ thị và các tệp JSON tổng hợp nhỏ.
- **GitHub Release / Zenodo:** Đóng gói toàn bộ `cifar10_10k_artifacts.tar.gz` (kèm DOI cho ấn bản khoa học) phục vụ lưu trữ vĩnh viễn và kiểm định ngang hàng (peer-review).
