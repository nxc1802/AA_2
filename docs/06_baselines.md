# 06. Phương Pháp Đối Thủ & Chuẩn So Sánh (Baselines Methodology)

> **Dự án:** AA (Sparse Adversarial Attack Benchmark)  
> **Phương pháp trung tâm:** CASA  
> **Trạng thái:** Canonical Documentation (V2 Specification)  
> **Mục tiêu:** Phân loại các đối thủ cạnh tranh, ghi nhận chi tiết nguồn gốc (provenance), tham số chuẩn, cơ chế bộ chuyển đổi (adapters) và nguyên tắc so sánh công bằng.

---

## 1. Phân Loại Đối Thủ Cạnh Tranh (Attack Taxonomy)

Để xây dựng một bức tranh so sánh khoa học đa chiều và khách quan, dự án phân chia các phương pháp tấn công thành 4 nhóm phương pháp luận rõ rệt:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                            HỆ THỐNG PHÂN LOẠI BASELINES                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. DENSE REFERENCES (Tham chiếu neo ngữ cảnh):                              │
│    • Can thiệp 100% điểm ảnh dưới chuẩn L_∞. Không xếp hạng trực tiếp với   │
│      tấn công thưa thớt, chỉ dùng làm mốc chặn trên về độ nhạy cảm của mạng.│
│    • Đại diện: FGSM (Goodfellow'15), BIM (Kurakin'17), PGD (Madry'18).      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. FIXED-BUDGET SPARSE ATTACKS (Cùng ngân sách cố định K):                  │
│    • Đối thủ trực diện của CASA. Tìm nhiễu đối nghịch thỏa mãn ||δ||_0 ≤ K. │
│    • Hộp trắng (White-box): SPGD (ICML'19), PGD0 (Croce'19).                │
│    • Hộp đen (Black-box): Sparse-RS (NeurIPS'20), CornerSearch (ECCV'20).   │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. MINIMAL-SUPPORT ATTACKS (Tìm kiếm tập hỗ trợ cực tiểu):                  │
│    • Tối thiểu hóa L_0 để bẻ gãy mô hình. Chạy theo công thức nguyên bản,   │
│      sau đó nội suy đường cong ASR@K dựa trên mức L_0 thực tế đạt được.     │
│    • Đại diện: SparseFool (CVPR'19), Sigma-Zero (NeurIPS'24), GSE (CVPR'22).│
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. COMPOSITE BENCHMARK:                                                     │
│    • Tập hợp đa giải thuật kết hợp: Sparse-AutoAttack (sAA).                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hồ Sơ Kỹ Thuật Chi Tiết Từng Baseline (Method Profiles)

### 2.1 Sparse-PGD / SPGD (White-Box Fixed-Budget)
- **Công bố:** *Sparse Adversarial Attack via Perturbation-Decoupled Direction and Norm* (ICML 2019)
- **Kho lưu trữ gốc:** `https://github.com/CityU-MLO/sPGD`
- **Pinned Commit:** `37564941d11a9a72c4c4fa2b07299f30dae26154`
- **Bộ chuyển đổi (Adapter):** [`src/aa/attacks/external/spgd.py`](file:///Volumes/WorkSpace/Project/AA/src/aa/attacks/external/spgd.py)
- **Phương pháp luận:** Tách rời hướng gradient và chuẩn $L_0$, sử dụng phép chiếu thưa thớt (sparse projection) sau mỗi bước PGD.
- **Tham số chuẩn:** `steps = 100`, `alpha = 0.05`.

### 2.2 Sparse-RS (Black-Box Random Search)
- **Công bố:** *Sparse-RS: a versatile framework for query-efficient sparse attacks via random search* (NeurIPS 2020)
- **Kho lưu trữ gốc:** `https://github.com/fra31/sparse-rs`
- **Pinned Commit:** `21d875969a1455e4d5b26dcf32c843e6262d1f9c`
- **Giấy phép gốc:** MIT License
- **Bộ chuyển đổi (Adapter):** [`src/aa/attacks/external/sparse_rs.py`](file:///Volumes/WorkSpace/Project/AA/src/aa/attacks/external/sparse_rs.py)
- **Phương pháp luận:** Tìm kiếm ngẫu nhiên có định hướng (Random Search), thay đổi các khối pixel nhỏ và kiểm tra hàm mục tiêu không cần đạo hàm.
- **Tham số chuẩn:** `n_queries = 10,000`, `alpha = 0.3`.

### 2.3 CornerSearch (Black-Box Fixed-Budget)
- **Công bố:** *Sparse and Imperceivable Adversarial Attacks* (ECCV 2020)
- **Kho lưu trữ gốc:** `https://github.com/fra31/sparse-imperceivable-attacks`
- **Pinned Commit:** `57c23ead05803a631c332be93f824ebd8020385d`
- **Bộ chuyển đổi (Adapter):** [`src/aa/attacks/external/cornersearch.py`](file:///Volumes/WorkSpace/Project/AA/src/aa/attacks/external/cornersearch.py)
- **Phương pháp luận:** Khảo sát giá trị hàm mục tiêu tại các đỉnh của siêu hộp $[0, 1]^3$ cho từng điểm ảnh để chọn tập pixel làm suy giảm độ tin cậy lớn nhất.
- **Tham số chuẩn:** `max_queries = 15,000`.

### 2.4 Sigma-Zero (White-Box Minimal-Support)
- **Công bố:** *Sigma-Zero: Gradient-based Sparse Adversarial Attacks with Exact L0-Budget Constraints* (NeurIPS 2024)
- **Kho lưu trữ gốc:** `https://github.com/sigma0-advx/sigma-zero`
- **Pinned Commit:** `f59494ca5fdb8f041e618381eb925e0bec00b01e`
- **Bộ chuyển đổi (Adapter):** [`src/aa/attacks/external/sigma_zero.py`](file:///Volumes/WorkSpace/Project/AA/src/aa/attacks/external/sigma_zero.py)
- **Phương pháp luận:** Nới lỏng hàm chỉ thị $L_0$ bằng hàm sigmoid có tham số nhiệt độ $\sigma$, tối ưu hóa trơn trước khi làm nguội (annealing) về giá trị rời rạc.
- **Tham số chuẩn:** `steps = 100`, `sigma_init = 1.0`, `sigma_final = 0.001`.

### 2.5 SparseFool (White-Box Geometric Minimal-Support)
- **Công bố:** *SparseFool: a few pixels make a big difference* (CVPR 2019)
- **Kho lưu trữ gốc:** `https://github.com/LTS4/SparseFool`
- **Pinned Commit:** `958e2bf663ea4b22ecc7a24dcf226a9fab77124f`
- **Giấy phép gốc:** Apache-2.0
- **Bộ chuyển đổi (Adapter):** [`src/aa/attacks/external/sparsefool.py`](file:///Volumes/WorkSpace/Project/AA/src/aa/attacks/external/sparsefool.py)
- **Phương pháp luận:** Mở rộng DeepFool sang chuẩn $L_1$ rồi chiếu trực giao về siêu phẳng quyết định để tìm tập điểm ảnh cực tiểu.
- **Tham số chuẩn:** `max_iterations = 20`, `lambda_ = 3.0`.

### 2.6 GSE (Geometric Sparse Exploration)
- **Công bố:** *Greedy Spatial Exploration for Sparse Adversarial Attacks* (CVPR 2022)
- **Kho lưu trữ gốc:** `https://github.com/wagnermoritz/GSE`
- **Pinned Commit:** `0123cdf6b88f9313f40886aa13631dc82ff43874`
- **Bộ chuyển đổi (Adapter):** [`src/aa/attacks/external/gse.py`](file:///Volumes/WorkSpace/Project/AA/src/aa/attacks/external/gse.py)

---

## 3. Lý Do Lựa Chọn & Loại Bỏ Các Phương Pháp (Inclusion & Exclusion Rationale)

Dự án thực hiện rà soát nghiêm ngặt các mã nguồn upstream nhằm bảo vệ tính toàn vẹn khoa học:

| Phương pháp | Quyết định | Cơ sở khoa học & Pháp lý |
| :--- | :---: | :--- |
| **SPGD, Sparse-RS, CornerSearch** | **BẮT BUỘC (Core)** | Các phương pháp kinh điển, được trích dẫn rộng rãi nhất trong mảng $L_0$. Mã nguồn chính thức đã được thẩm định tính toán vẹn. |
| **Sigma-Zero** | **BẮT BUỘC (SOTA '24)** | Đại diện mới nhất (NeurIPS 2024) cho hướng tiếp cận nới lỏng liên tục. |
| **SparseFool, GSE** | **BẮT BUỘC (Min-Support)** | Cung cấp góc nhìn về mức $L_0$ cực tiểu cần thiết để bẻ gãy mô hình. |
| **Homotopy (SparseADV)** | **TẠM LOẠI BỎ (Unbundled)** | Upstream repository không xác định được commit ổn định, phụ thuộc nhiều thư viện C++ cũ không tương thích PyTorch 2.x hiện đại. |
| **SAIF** | **LOẠI BỎ HOÀN TOÀN** | Mã nguồn tác giả không công khai chính thức; các bản cài đặt trôi nổi là placeholder không thể kiểm chứng tính khoa học. |

---

## 4. Nguyên Tắc So Sánh Công Bằng (Fair Comparison Protocol)

Để tránh hiện tượng "đánh giá thiên vị" (strawman baselines):
1. **Khớp đúng ngân sách ($K$):** Tất cả các phương pháp Fixed-budget đều nhận cùng một danh sách $K \in \{1, 2, 4, 8, 16, 32, 64\}$.
2. **Nội suy chuẩn xác cho Minimum-Support:** Đối với SparseFool, Sigma-Zero, GSE (vốn giải bài toán tìm $\min \|\delta\|_0$), một mẫu được tính là thành công tại mốc $K$ nếu và chỉ nếu:
   $$\|\delta_{\text{achieved}}\|_{0, \text{spatial}} \le K \quad \text{và} \quad f(x + \delta) \neq y$$
3. **Phân định rạch ròi chi phí:** Không bao giờ tuyên bố thuật toán hộp trắng "nhanh hơn" thuật toán hộp đen chỉ dựa vào số lượt query, mà phải báo cáo đồng thời $F + 2B$ (hộp trắng) và số queries (hộp đen).
4. **Không sửa đổi lõi thuật toán:** Mọi bộ chuyển đổi (Adapter) trong `src/aa/attacks/external/` chỉ đóng vai trò chuyển đổi định dạng dữ liệu (PyTorch tensor $\leftrightarrow$ NumPy array), tuyệt đối không can thiệp vào hàm mục tiêu, luật cập nhật hay điều kiện dừng của tác giả gốc.
