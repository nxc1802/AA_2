# Comprehensive Benchmarking and Defense Analysis for Sparse Adversarial Attacks on Deep Neural Networks

> **Subtitle:** Benchmarking Sparse Attacks, Understanding Their Robustness, and Developing a Novel Sparse Attack Framework
> 
> * **Created At:** `2026-08-04T11:48:43Z`
> * **Completed At:** `2026-08-04T11:48:43Z`
> * **File Path:** [`docs/plan.md`](file:///Volumes/WorkSpace/Project/AA/docs/plan.md)

---

## 1. Motivation

Trong nhiều năm qua, **adversarial attack** chủ yếu tập trung vào các ràng buộc norm:
* **$L_\infty$**: FGSM, BIM, PGD
* **$L_2$**: C&W, DeepFool

Tuy nhiên, các perturbation này thường thay đổi toàn bộ pixel trên bức ảnh.

Trong thực tế, tồn tại một hướng nghiên cứu rất thú vị: **chỉ thay đổi rất ít pixel nhưng vẫn fool được model.**  
Đây chính là **Sparse Adversarial Attack**.

### Phân loại các phương pháp SOTA:
* **Classical:** JSMA, One Pixel, SparseFool, CornerSearch
* **Optimization-based:** SAIF, $\sigma$-zero, Homotopy, PGD0, GSE, Sparse-PGD
* **Attention / Attribution:** IPFSA, Gradient Guidance
* **Frequency-aware:** SFA
* **Black-box:** Sparse-RS, BruSLeAttack, Pixle

> [!NOTE]
> Các nghiên cứu gần đây còn mở rộng sparse perturbation theo hướng có cấu trúc (*structured sparsity*), cơ chế chú ý (attention/attribution), tần số (frequency-aware) hoặc tối ưu hóa hiệu quả hơn, cho thấy đây vẫn là một hướng nghiên cứu còn rất “mở”.

---

## 2. Research Questions

Paper sẽ giải đáp 6 câu hỏi nghiên cứu chính (**RQs**):

* **RQ1:** Sparse attack nào mạnh nhất trong số các họ thuật toán (Classical, Optimization, Attention, Frequency, Black-box)?
* **RQ2:** Sparse attack transfer tốt hơn dense attack hay không?
* **RQ3:** Defense hiện nay chống sparse attack tốt đến đâu?
* **RQ4:** Preprocessing có hiệu quả không?
* **RQ5:** Adversarial Training chống sparse attack tốt không?
* **RQ6:** Có thể thiết kế sparse attack mới mạnh hơn không? *(Proposed Method — Xem tài liệu chi tiết tại [`proposed_method.md`](file:///Volumes/WorkSpace/Project/AA/docs/proposed_method.md))*

---

## 3. Scope

Paper được chia thành **4 phần lớn**:

* **Part A:** Attack Benchmark (Bao gồm các phương pháp SOTA mới cập nhật)
* **Part B:** Defense Benchmark
* **Part C:** Defense Analysis
* **Part D:** Proposed Sparse Attack *(Chi tiết thiết kế và phân tích novelty đã được tách sang tài liệu [`proposed_method.md`](file:///Volumes/WorkSpace/Project/AA/docs/proposed_method.md))*

> [!IMPORTANT]
> Trong đó, **Proposed Method** hoàn toàn độc lập với các phần benchmark và được quản lý riêng để linh hoạt cải tiến.

---

## PART A: Attack Benchmark

### Baseline Attacks (Dense Perturbation)

| Attack | Norm Constraint | Characteristics / Steps |
| :--- | :---: | :--- |
| **FGSM** | $L_\infty$ | 1 step |
| **BIM** | $L_\infty$ | Iterative FGSM |
| **PGD** | $L_\infty$ | Strong baseline |
| **DeepFool** *(optional)* | $L_2$ | Geometry-based |
| **C&W** *(optional)* | $L_2$ | Optimization-based |

* **Mục tiêu:** So sánh **Dense perturbation** vs **Sparse perturbation**.

---

### SOTA Sparse Attack Benchmark Taxonomy

Cấu trúc phân loại các Sparse Attacks trong bộ benchmark SOTA:

```text
Sparse Adversarial Attacks
│
├── Classical Sparse Attacks
│   ├── JSMA (Papernot, 2016)
│   ├── One Pixel Attack (Su et al., 2019)
│   ├── SparseFool (Modas et al., 2019)
│   └── CornerSearch (Croce & Hein, 2019)
│
├── Optimization-based Sparse Attacks
│   ├── SAIF
│   ├── σ-zero (Sigma-zero)
│   ├── Homotopy
│   ├── PGD0 (PGD-L0)
│   ├── GSE (Group-wise Sparse Attack, ICLR 2025)
│   └── Sparse-PGD (2024)
│
├── Attention / Attribution Sparse Attacks
│   ├── IPFSA
│   └── Gradient Guidance
│
├── Frequency-aware Sparse Attacks
│   └── SFA (Sparse Frequency Attack)
│
└── Black-box Sparse Attacks
    ├── Sparse-RS (Croce et al., 2022)
    ├── BruSLeAttack
    └── Pixle (Pixel rearrangement)
```

---

### Chi Tiết Phân Loại Các Phương Pháp Sparse Attacks (SOTA)

| Phân loại (Category) | Phương pháp (Attack Method) | Xuất xứ / Đặc trưng nổi bật (Key Characteristics) |
| :--- | :--- | :--- |
| **Classical Sparse Attacks** | **JSMA** | Saliency Map, Papernot (2016) |
| | **One Pixel Attack** | Differential Evolution, Black-box, Su et al. (2019) |
| | **SparseFool** | Geometry-based, Fast, Efficient, Modas et al. (2019) |
| | **CornerSearch** | Greedy Search trên các vị trí góc/biên, Croce & Hein (2019) |
| **Optimization-based Sparse Attacks** | **SAIF** | Sparse Attack via Iterative Filtering / Optimization |
| | **$\sigma$-zero** | Tối ưu hóa xấp xỉ liên tục cho chuẩn $L_0$ |
| | **Homotopy** | Homotopy optimization cho bài toán thưa |
| | **PGD0 ($PGD-L_0$)** | PGD kết hợp chiếu không gian thưa chuẩn $L_0$ |
| | **GSE** | Group-wise Sparse Attack (ICLR 2025) |
| | **Sparse-PGD** | Unified sparse framework (2024) |
| **Attention / Attribution Sparse Attacks**| **IPFSA** | Integrated Gradient / Attribution-guided Pixel Selection |
| | **Gradient Guidance** | Định hướng vùng tấn công dựa trên Attention / Saliency Map |
| **Frequency-aware Sparse Attacks** | **SFA** | Sparse Frequency Attack (Tấn công thưa trên miền tần số FFT/DCT) |
| **Black-box Sparse Attacks** | **Sparse-RS** | Random Search chuyên biệt cho $L_0$ perturbation (Croce et al., 2022) |
| | **BruSLeAttack** | Black-box Score-based / Decision-based Sparse Attack |
| | **Pixle** | Pixel rearrangement attack, Black-box |

---

### Experimental Setup

#### 1. Datasets
* **CIFAR10**: Mandatory
* **CIFAR100**: Recommended
* **TinyImageNet**: Recommended
* **ImageNet-100**: Optional

#### 2. Models (Backbones)
* **ResNet18**
* **ResNet50**
* **WideResNet**
* **ViT-B** *(optional)*

#### 3. Evaluation Metrics
* **Accuracy & Effectiveness:**
  * Natural Accuracy
  * Attack Success Rate (ASR)
  * Robust Accuracy
  * Fooling Rate
* **Efficiency & Resource:**
  * Query Number *(for black-box attacks)*
  * Runtime
* **Perturbation & Distance:**
  * Number of perturbed pixels
  * $L_0$ distance
  * $L_2$ distance
  * $L_\infty$ distance
* **Visual Quality:**
  * SSIM
  * PSNR
  * LPIPS
* **Transferability:**
  * Source Model $\rightarrow$ Target Model

#### 4. Output Format Example

| Category | Attack | ASR | $L_0$ | $L_2$ | Queries | Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Classical | JSMA | ... | ... | ... | N/A | ... |
| Classical | One Pixel | ... | ... | ... | ... | ... |
| Classical | CornerSearch | ... | ... | ... | ... | ... |
| Optimization | SAIF | ... | ... | ... | N/A | ... |
| Optimization | $\sigma$-zero | ... | ... | ... | N/A | ... |
| Optimization | Homotopy | ... | ... | ... | N/A | ... |
| Optimization | PGD0 | ... | ... | ... | N/A | ... |
| Attention | IPFSA | ... | ... | ... | N/A | ... |
| Frequency | SFA | ... | ... | ... | N/A | ... |
| Black-box | Sparse-RS | ... | ... | ... | ... | ... |
| Black-box | BruSLeAttack | ... | ... | ... | ... | ... |

---

## PART B: Defense Benchmark

> [!NOTE]
> Đây là phần nhiều paper bỏ qua. Paper này nên benchmark toàn diện.

### Defense Category 1: Preprocessing Defense

* **Bao gồm:**
  * Gaussian Blur
  * Median Filter
  * JPEG Compression
  * Bit Depth Reduction
  * Random Resize
  * Random Crop
  * TVM (Total Variation Minimization)
  * Wavelet Denoising
  * Pixel Deflection
  * Feature Squeezing
  * Non-local Means

* **Pipeline Đánh giá:**  
  $$\text{Attack} \longrightarrow \text{Defense} \longrightarrow \text{Model}$$

* **Metrics:**
  * Robust Accuracy
  * Recovery Rate
  * Attack Reduction

---

### Defense Category 2: Input Transformation

* Randomization
* Resize-Padding
* Random Noise
* Frequency Filter

---

### Defense Category 3: Detection *(Optional)*

* Feature-based
* Confidence-based

---

### Defense Category 4: Adversarial Training

> [!IMPORTANT]
> Đây là phần quan trọng nhất.

* **Benchmark Pipeline:**  
  $$\text{Standard Training} \longrightarrow \text{FGSM AT} \longrightarrow \text{PGD AT} \longrightarrow \text{TRADES} \longrightarrow \text{MART} \longrightarrow \text{Sparse Adversarial Training (nếu có)}$$

> Các nghiên cứu trước cho thấy adversarial training chuẩn cải thiện khả năng chống perturbation dày đặc, nhưng hiệu quả với sparse attacks còn hạn chế; các framework gần đây như Sparse-PGD đề xuất adversarial training chuyên biệt cho sparse perturbations.

* **Evaluation Metrics:**  
  $$\text{Clean Accuracy} \longrightarrow \text{Robust Accuracy} \longrightarrow \text{Sparse Robust Accuracy} \longrightarrow \text{Generalization}$$

---

## PART C: Analysis

> [!TIP]
> Đây là phần rất quan trọng. Paper nên có rất nhiều figure minh họa.

1. **Analysis 1: Perturbed Pixel Distribution** — Heatmap
2. **Analysis 2: GradCAM** — Sparse perturbation đánh vào đâu?
3. **Analysis 3: Frequency Analysis** — FFT
4. **Analysis 4: Layer Sensitivity** — Perturb layer $\rightarrow$ Output
5. **Analysis 5: Transferability**
6. **Analysis 6: Defense Failure** — Tại sao defense fail?
7. **Analysis 7: Runtime**
8. **Analysis 8: Visual Comparison** — Original $\rightarrow$ FGSM $\rightarrow$ PGD $\rightarrow$ Classical/Opt/Attn/Freq/Blackbox $\rightarrow$ Proposed
9. **Analysis 9: Sparsity vs ASR** — Trade-off Curve
10. **Analysis 10: Perturbation Budget** — $L_0$ vs ASR

---

## PART D: Proposed Method

> [!NOTE]
> Phần **Proposed Method** đã được tách ra một tài liệu quản lý riêng để dễ dàng cập nhật và nghiên cứu chuyên sâu.  
> Chi tiết thiết kế, so sánh với GSE, 4 hướng đề xuất (CPA, FCSA, FMSA, HSA) và chiến lược được chọn có thể tham khảo trực tiếp tại:
> 
> 👉 **[`docs/proposed_method.md`](file:///Volumes/WorkSpace/Project/AA/docs/proposed_method.md)**

---

## Project Structure

```text
SparseAA/
├── datasets/
├── models/
├── attacks/
│   ├── baselines/
│   │   ├── fgsm.py
│   │   ├── bim.py
│   │   └── pgd.py
│   ├── classical/
│   │   ├── jsma.py
│   │   ├── onepixel.py
│   │   ├── sparsefool.py
│   │   └── corner_search.py
│   ├── optimization/
│   │   ├── saif.py
│   │   ├── sigma_zero.py
│   │   ├── homotopy.py
│   │   ├── pgd0.py
│   │   ├── gse.py
│   │   └── sparse_pgd.py
│   ├── attention_attribution/
│   │   ├── ipfsa.py
│   │   └── gradient_guidance.py
│   ├── frequency/
│   │   └── sfa.py
│   ├── blackbox/
│   │   ├── sparse_rs.py
│   │   ├── brusle.py
│   │   └── pixle.py
│   └── proposed/                 # Implement Proposed Method (FCSA / CPA)
├── defenses/
│   ├── preprocessing/
│   ├── adversarial_training/
│   ├── transforms/
│   └── detection/
├── benchmark/
├── visualization/
├── analysis/
├── experiments/
└── paper/
```

---

## Roadmap thực hiện

| Giai đoạn | Nội dung | Kết quả |
| :--- | :--- | :--- |
| **Phase 1** | Cài đặt các bộ benchmark SOTA (Classical: JSMA, One Pixel, SparseFool, CornerSearch; Optimization: SAIF, $\sigma$-zero, Homotopy, PGD0, GSE, Sparse-PGD; Attention: IPFSA, Gradient Guidance; Frequency: SFA; Black-box: Sparse-RS, BruSLe, Pixle) | Bộ benchmark chuẩn SOTA toàn diện |
| **Phase 2** | Đánh giá trên CIFAR-10/100, TinyImageNet với nhiều backbone (ResNet18, ResNet50, WideResNet, ViT-B) | Bảng kết quả baseline & SOTA toàn diện |
| **Phase 3** | Đánh giá defense (preprocessing + adversarial training) đối với các họ sparse attack | Benchmark defense toàn diện |
| **Phase 4** | Phân tích sâu (ASR, $L_0$, transferability, visualization, runtime, Grad-CAM, FFT, sparsity trade-off) | Phần analysis cho paper |
| **Phase 5** | Phát triển và đánh giá Proposed Method (Tham khảo [`proposed_method.md`](file:///Volumes/WorkSpace/Project/AA/docs/proposed_method.md)) | Đóng góp chính của paper |

---

## Đánh giá tổng quan

> Theo đánh giá của tôi, cấu trúc này có thể tạo ra một paper mạnh vì nó không chỉ đề xuất một thuật toán mới mà còn xây dựng một benchmark toàn diện đầu tiên (hoặc gần như toàn diện) cho sparse adversarial attacks kết hợp đánh giá defense. Phần benchmark và defense sẽ tạo nền tảng ổn định, trong khi Proposed Method được giữ độc lập để có thể thay đổi, cải tiến hoặc thay thế mà không ảnh hưởng tới toàn bộ pipeline nghiên cứu.