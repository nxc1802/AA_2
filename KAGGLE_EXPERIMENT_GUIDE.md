# Hướng Dẫn Thực Nghiệm Toàn Diện Trên Kaggle GPU (`KAGGLE_EXPERIMENT_GUIDE.md`)

Tài liệu này cung cấp quy trình từng bước để thực hiện toàn bộ các thực nghiệm tính toán lớn trên **Kaggle GPU**, đáp ứng 100% tiêu chuẩn trong `checklist.md` để hoàn thiện công bố khoa học (Paper-Ready).

---

## 1. Mục Tiêu Thực Nghiệm & Phân Bổ Thời Gian (GPU Budget)

### Tận dụng 2 GPU T4 (Multi-GPU Sharding)
Kaggle cung cấp **2x GPU Tesla T4 (16GB VRAM mỗi card)**. Do bài toán đánh giá tấn công đối nghịch là **độc lập tuyệt đối giữa các mẫu ảnh (Embarrassingly Parallel)**:
- Codebase sử dụng `MultiGPUScheduler` tự động chia đều tập 10,000 ảnh cho 2 GPU chạy song song (GPU 0: 5,000 ảnh; GPU 1: 5,000 ảnh).
- Kết quả thu được **chính xác 100% và bảo toàn toàn bộ tính chất toán học** so với chạy 1 GPU, nhưng **thời gian thực thi giảm gần một nửa (~1.9x speedup)**.

### Tối ưu hóa Batch Size
- **Đối với Baselines (SPGD, Sigma-Zero, Sparse-RS, FGSM, PGD)**: Tăng lên `batch_size = 512` (hoặc `256`) trên mỗi card để tận dụng tối đa số nhân CUDA của T4.
- **Đối với CASA (Đề xuất)**: Tăng từ `16` lên **`batch_size = 32`** (hoặc `64`). Thí nghiệm `test_bs_tradeoff.py` cho thấy throughput tăng từ 5.3 img/s lên **6.5 – 7.2 img/s** (tăng thêm ~35% tốc độ) mà VRAM chỉ tốn ~580MB – 1.1GB, hoàn toàn nằm trong giới hạn 16GB của T4.

| Giai đoạn (Stage) | Nội dung thực nghiệm | Quy mô dữ liệu | Thời gian (1x T4) | Thời gian (2x T4 Song Song) | Mức độ ưu tiên |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Stage 0: Smoke Gate** | Kiểm thử kết nối & pipeline | 20 mẫu CIFAR-10 | ~1 – 2 phút | ~1 phút | **Bắt buộc** |
| **Stage 1: CASA 10k** | CASA $K \in \{1, 2, 4, 8, 16, 32, 64\}$ | **10,000 mẫu test** | ~45 – 60 phút | **~25 – 35 phút** | **P0 (Cốt lõi Paper)** |
| **Stage 2: Strong Baselines** | SPGD (100 st) & Sigma-Zero (500 st) | **10,000 mẫu test** | ~35 – 50 phút | **~20 – 25 phút** | **P0 (So sánh SOTA)** |
| **Stage 3: Sparse-RS 10k** | Sparse-RS (10,000 queries) | **10,000 mẫu test** | ~3 – 3.5 giờ | **~1.5 – 1.8 giờ** | **P0 (Baseline Blackbox)** |
| **Stage 4: Ablation Study** | 10 biến thể bóc tách cơ chế CASA | 1,000 mẫu ($K=1, 4, 16$) | ~35 – 45 phút | **~18 – 25 phút** | **P0 (Chứng minh đóng góp)** |
| **Stage 5: Failure Diagnostics**| Trích xuất phân tích mẫu lỗi | 1,000 mẫu ($K=1, 2, 4$) | ~15 phút | **~8 phút** | **P1 (Diagnostic Insights)** |
| **Stage 6: Figures & Report** | Tự động vẽ 6 đồ thị & xuất báo cáo | Toàn bộ artifacts | ~2 phút | ~2 phút | **P1 (Paper Artifacts)** |

> [!TIP]
> **Tổng thời gian chạy toàn bộ trên 2x GPU T4 chỉ còn khoảng ~2.5 – 3.2 giờ** (thay vì 5.5 – 6 giờ trên 1 GPU). Bạn sẽ tiết kiệm được hơn nửa thời gian và hoàn toàn an tâm không bao giờ chạm ngưỡng timeout 12h của Kaggle!


---

## 2. Chuẩn Bị Môi Trường Trên Kaggle

### Bước 2.1: Tạo Kaggle Notebook mới
1. Truy cập [Kaggle Code](https://www.kaggle.com/code) $\to$ Chọn **New Notebook**.
2. Tại bảng điều khiển bên phải (**Notebook Settings**):
   - **Accelerator**: Chọn **GPU T4 x2** (hoặc **GPU P100**).
   - **Internet**: Chuyển sang **On** *(Bắt buộc bật để tải checkpoint và dataset)*.
   - **Persistence**: Chọn **Files only** (hoặc **Variables and Files**).

### Bước 2.2: Đưa mã nguồn lên Kaggle

Có 2 cách đơn giản nhất:

#### Cách A: Tải file Notebook `kaggle_paper_runner.ipynb` lên Kaggle (Khuyên dùng)
1. Trong giao diện Kaggle Notebook: Vào menu **File** $\to$ **Import Notebook** $\to$ Chọn file [`kaggle_paper_runner.ipynb`](file:///Volumes/WorkSpace/Project/AA/kaggle_paper_runner.ipynb) từ máy của bạn.
2. Nén mã nguồn local thành file zip:
   ```bash
   cd /Volumes/WorkSpace/Project/AA
   zip -r aa_source.zip . -x "*.git*" "*__pycache__*" "*test_attack_cache*"
   ```
3. Tạo một **New Dataset** trên Kaggle, tải file `aa_source.zip` lên, sau đó đính kèm Dataset này vào Notebook.

#### Cách B: Git clone trực tiếp từ GitHub (Nếu repo đã push)
Trong ô code đầu tiên của Kaggle Notebook:
```bash
!git clone https://github.com/<username>/<repo_name>.git aa
%cd aa
```

---

## 3. Quy Trình Chạy Thực Nghiệm Từng Bước

Nếu chạy qua giao diện dòng lệnh bash trên terminal của Kaggle hoặc background session:

```bash
# Cài đặt môi trường & thư viện
pip install -r requirements.txt
pip install -e .

# Chạy toàn bộ pipeline tự động
bash scripts/kaggle_run.sh all
```

Hoặc chạy từng bước độc lập theo các lệnh sau:

### Lệnh 1: Kiểm thử nhanh Smoke Test
Đảm bảo mọi cấu hình và adapter hoạt động trơn tru:
```bash
python3 scripts/attack_benchmark.py --config configs/smoke.yaml --strict --output result/smoke_results.json
```

### Lệnh 2: Chạy CASA trên 10,000 ảnh CIFAR-10 (Claim Paper SOTA)
```bash
python3 scripts/run_casa_benchmark.py \
    --samples 10000 \
    --batch-size 16 \
    --k-values 1 2 4 8 16 32 64 \
    --output result/casa_10000_results.json
```
*Kết quả sẽ ghi nhận ASR@K, Wilson 95% Confidence Interval, và thời gian thực thi.*

### Lệnh 3: Chạy Baselines Đối Thủ (SPGD & Sigma-Zero) trên 10k
```bash
python3 scripts/attack_benchmark.py \
    --config configs/paper_cifar10.yaml \
    --attacks spgd,sigma_zero \
    --output result/baselines_spgd_sigmazero_10k.json
```

### Lệnh 4: Chạy Sparse-RS (10k queries) trên 10k
```bash
python3 scripts/attack_benchmark.py \
    --config configs/paper_cifar10.yaml \
    --attacks sparse_rs \
    --output result/sparse_rs_10k.json
```

### Lệnh 5: Chạy Nghiên Cứu Phân Rã Thành Phần (Ablation Study)
```bash
python3 scripts/run_ablation.py \
    --samples 1000 \
    --k-values 1 4 16 \
    --output result/ablation_results.json \
    --report-md docs/ablation_study.md
```

### Lệnh 6: Phân Tích Các Trường Hợp Thất Bại (Failure Diagnostics)
```bash
python3 scripts/analyze_failures.py \
    --samples 1000 \
    --k-values 1 2 4 \
    --max-failures 30 \
    --output-json result/failure_analysis.json \
    --output-md docs/failure_analysis.md
```

### Lệnh 7: Tạo Bộ Biểu Đồ Chuẩn Publication & Xuất Báo Cáo
```bash
python3 scripts/plot_paper_figures.py --output-dir result/figures
```

---

## 4. Danh Mục Kết Quả & Đóng Gói Bài Báo (Deliverables)

Sau khi hoàn thành trên Kaggle, chạy lệnh nén sau để tải toàn bộ kết quả về máy:
```bash
tar -czvf paper_artifacts.tar.gz result/ docs/
```

Các file nhận được sẽ dùng để điền trực tiếp vào bài báo:

1. **`result/figures/figure2_asr_vs_k.pdf`**: Biểu đồ chính của bài báo ($ASR@K$ vs $K$) so sánh CASA với toàn bộ đối thủ.
2. **`result/figures/figure3_actual_l0_vs_k.png`**: Minh chứng Drop-and-Repair nén $L_0$ thực tế xuống thấp hơn budget $K$.
3. **`result/figures/figure4_efficiency_frontier.png`**: Biểu đồ đánh đổi Hiệu năng vs Thời gian tính toán (Efficiency Frontier).
4. **`result/figures/figure5_ablation_progression.png`**: Biểu đồ cột phân rã đóng góp của từng cơ chế.
5. **`docs/ablation_study.md`**: Bảng dữ liệu Ablation sẵn sàng copy sang bảng LaTeX.
6. **`docs/failure_analysis.md`**: Phân tích định tính (Qualitative Analysis) cho phần Discussion của Paper.
7. **`result/casa_10000_results.json`**: Chứa raw data, ASR kèm 95% Confidence Interval cho toàn bộ 10,000 ảnh test.

---

## 5. Xử Lý Các Sự Cố Thường Gặp (Troubleshooting)

* **Hết bộ nhớ GPU (CUDA Out-of-Memory)**:
  - Nếu gặp OOM khi chạy CASA, giảm batch size từ 16 xuống 8: `--batch-size 8`.
* **Kaggle Kernel bị ngắt kết nối (Disconnect)**:
  - Sử dụng tính năng **Save & Run All (Commit)** ở góc trên bên phải màn hình Kaggle. Notebook sẽ chạy ngầm trên server của Kaggle ngay cả khi bạn tắt trình duyệt hoặc mất kết nối internet.
* **Checkpoint không tải được**:
  - Script đã tích hợp hàm tự động tải từ Hugging Face Hub (`Cuong2004/AA`). Hãy đảm bảo tùy chọn **Internet: On** đã được bật trong Settings của Kaggle.
