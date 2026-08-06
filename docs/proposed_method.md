# Proposed Method Design & Novelty Analysis: Sparse Adversarial Attacks

> **Subtitle:** Novel Formulation and Comparative Analysis for High-Impact Paper Contributions
> 
> * **Created At:** `2026-08-04T12:28:52Z`
> * **Completed At:** `2026-08-04T12:28:52Z`
> * **File Path:** [`docs/proposed_method.md`](file:///Volumes/WorkSpace/Project/AA/docs/proposed_method.md)

---

## 1. Tổng Quan 4 Hướng Thiết Kế Proposed Method

Các hướng thiết kế dưới đây được sắp xếp theo thứ tự ưu tiên từ **“ít rủi ro - dễ triển khai”** đến **“rất mới nhưng thách thức cao”**:

```text
Option A: CPA (Cooperative Pixels Attack)
  └─► [Rủi ro thấp - Dễ làm]

Option B: FCSA (Functional Coalition Sparse Attack)
  └─► [Rủi ro thấp - Học thuật mạnh]

Option C: FMSA (Feature-to-Minimal Support Attack)
  └─► [Rủi ro trung bình - Ý tưởng đột phá]

Option D: HSA (Hypergraph Sparse Attack)
  └─► [Rủi ro cao - Rất tham vọng]
```

---

### Option A — Cooperative Pixels Attack (CPA)

#### Core Idea (Ý tưởng cốt lõi)
* Các phương pháp Sparse Attack hiện nay giả định: **Các pixel được lựa chọn độc lập**.
* **CPA thay đổi giả định đó:** Pixel được lựa chọn dựa trên đóng góp phối hợp (*cooperative contribution*).

```text
Quy trình hiện nay:
Pixel ──► Importance ──► Top-k

Quy trình đề xuất CPA:
Pixel ──► Interaction Score ──► Coalition Score ──► Sparse Optimization
```

#### Novelty (Tính mới)
* Không thay đổi optimizer.
* Không thay đổi loss function.
* **Chỉ thay đổi cách lựa chọn support:** Support được xây dựng dựa trên sự phối hợp (*cooperation*).

#### Kỹ thuật xây dựng Cooperation Score:
Có thể sử dụng các chỉ số:
* Gradient correlation
* Representation similarity
* Activation dependency
* Mutual influence

#### Ưu & Nhược điểm:
* **Ưu điểm:** Dễ cài đặt (*implement*), dễ benchmark, dễ kết hợp với các thuật toán tối ưu $L_0$ như $\sigma$-zero.
* **Nhược điểm:** Reviewer sẽ chất vấn: *"Cooperation được định nghĩa toán học cụ thể như thế nào?"* — Đây là điểm cần chuẩn bị lập luận cực kỳ chặt chẽ.

---

### Option B — Functional Coalition Sparse Attack (FCSA)

#### Core Idea (Ý tưởng cốt lõi)
Đây là phiên bản được phát triển mang tính học thuật cao hơn. Không còn tiếp cận dưới dạng từng **pixel** đơn lẻ mà tiếp cận dưới dạng **coalition (liên minh)**.

> **Định nghĩa Coalition:** Một coalition không phải chỉ là tập hợp pixel thông thường, mà là **minimal set of pixels** mà *chỉ khi cùng xuất hiện* mới gây ra hiện tượng phá hủy đại diện đặc trưng (*feature collapse*).

```text
Ví dụ minh họa:
Pixel A ────────► Chưa đủ phá representation
Pixel B ────────► Chưa đủ phá representation
Pixel A + B ────► Feature Collapse (Liên minh chức năng)
```

#### Objective (Hàm mục tiêu)
Không tối ưu hóa điểm tầm quan trọng của từng pixel đơn lẻ (*pixel importance*), mà tối ưu hóa điểm ảnh hưởng của liên minh (*coalition influence*):

$$\text{Score}(S) = \Delta F(S) - \sum_{i \in S} \Delta F(i)$$

*Trong đó:* $S$ là tập hợp coalition đại diện cho nhóm pixel hợp tác.

#### Novelty (Tính mới)
Bài báo không còn đơn thuần là một *"Sparse Attack"* thông thường mà được phát biểu lại (*reformulate*) thành bài toán: **Coalition Discovery Sparse Attack**.

#### Ưu & Nhược điểm:
* **Ưu điểm:** Rất dễ phát biểu đóng góp (*contribution*) trong bài báo:
  > *"We reformulate sparse attack as a coalition discovery problem."*
* **Nhược điểm:** Cần thiết kế thuật toán tối ưu coalition một cách hiệu quả.

---

### Option C — Feature-to-Minimal Support Attack (FMSA)

#### Core Idea (Ý tưởng cốt lõi)
Đây là hướng tiếp cận đảo ngược hoàn toàn quy trình truyền thống:

```text
Quy trình truyền thống (Literature):
Pixel ────────► Feature

Quy trình đảo ngược của FMSA:
Feature ──────► Minimal Pixel Support
```

#### Ý tưởng chi tiết:
1. Chọn đại diện đặc trưng quan trọng (*critical representation*), ví dụ: *penultimate feature*.
2. Đặt câu hỏi: *"Muốn làm cho đặc trưng này hoàn toàn biến mất thì cần tập hợp pixel tối thiểu (minimal support) là bao nhiêu?"*
3. Tức là **không tìm pixel**, mà **tìm minimal support**.

#### Pipeline thực hiện:
$$\text{Feature Importance} \longrightarrow \text{Critical Representation} \longrightarrow \text{Minimal Support Search} \longrightarrow \text{Sparse Attack}$$

#### Novelty & Đánh giá:
* **Novelty:** Sparse attack được định nghĩa trên không gian đại diện (*representation*), thay vì trên không gian pixel.
* **Ưu điểm:** Tạo sự khác biệt rất rõ ràng so với toàn bộ tài liệu nghiên cứu hiện có.
* **Nhược điểm:** Phải chứng minh được mặt lý thuyết/thực nghiệm: *Feature nào thực sự là critical representation*.

---

### Option D — Hypergraph Sparse Attack (HSA)

#### Core Idea (Ý tưởng cốt lõi)
Đây là hướng đi tham vọng nhất (*most ambitious*). Thay vì biểu diễn bằng đồ thị thông thường (*graph*), phương pháp sử dụng **Hypergraph**.

#### Định nghĩa Hypergraph:
* **Node (Đỉnh):** Tương ứng với từng pixel $i$.
* **Hyperedge (Siêu cạnh):** Tương ứng với một đại diện đặc trưng (*representation*).  
  *Ví dụ:* Feature channel 51 phụ thuộc vào 100 pixels $\Rightarrow$ 100 pixels này cùng nằm trên 1 Hyperedge.

```text
Biểu diễn toàn bộ ảnh ──► Hypergraph Structure
Mechanisms of Attack   ──► Tìm Minimum Coalition để phá vỡ nhiều Hyperedge nhất
```

#### Pipeline thực hiện:
$$\text{Image} \longrightarrow \text{Representation Analysis} \longrightarrow \text{Hypergraph Construction} \longrightarrow \text{Coalition Search} \longrightarrow \text{Sparse Perturbation}$$

#### Novelty & Đánh giá:
* **Novelty:** Chuyển dịch bài toán **Sparse Attack $\rightarrow$ Graph Learning / Hypergraph Optimization**.
* **Ưu điểm:** Phát biểu bài toán (*problem formulation*) hoàn toàn mới lạ.
* **Nhược điểm:** 
  * Rất khó lập trình cài đặt (*implement*).
  * Reviewer chắc chắn sẽ đặt câu hỏi: *"Hypergraph được xây dựng như thế nào và tại sao lại cần Hypergraph?"*

---

## 2. Bảng So Sánh Đánh Giá Các Phương Pháp

| Phương pháp (Method) | Tính mới (Novelty) | Độ khó Cài đặt | Mức độ Rủi ro |
| :--- | :---: | :---: | :---: |
| **Option A: Cooperative Pixels Attack (CPA)** | ★★★★☆ | ★★☆☆☆ | Thấp |
| **Option B: Functional Coalition Sparse Attack (FCSA)** | ★★★★★ | ★★★☆☆ | Thấp |
| **Option C: Feature-to-Minimal Support (FMSA)** | ★★★★★ | ★★★★☆ | Trung bình |
| **Option D: Hypergraph Sparse Attack (HSA)** | ★★★★★ | ★★★★★ | Cao |