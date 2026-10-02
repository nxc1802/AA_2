# 13. Giới Hạn Nghiên Cứu & Tuyên Bố Bị Giới Hạn (Limitations & Negative Scope)

> **Dự án:** AA (Sparse Adversarial Attack Benchmark)  
> **Phương pháp trung tâm:** CASA  
> **Trạng thái:** Authoritative Boundary Statement  
> **Mục tiêu:** Định hình minh bạch các ranh giới khoa học, điều kiện biên, các điểm hạn chế cố hữu của thuật toán và tuyên bố rõ ràng những kết luận **không được hỗ trợ** nhằm tránh hiện tượng khẳng định quá mức (overclaiming).

---

## 1. Giới Hạn Thuật Toán (Algorithmic & Methodological Limitations)

### 1.1 Chi phí tổ hợp khi mở rộng độ phân giải siêu cao (Resolution Scaling)
CASA được thiết kế tối ưu trên các kích thước ảnh vừa và nhỏ ($32 \times 32$ và $64 \times 64$, tương ứng $1.024 \to 4.096$ điểm ảnh). Khi mở rộng sang độ phân giải tiêu chuẩn của ImageNet ($224 \times 224 = 50.176$ điểm ảnh) hoặc ảnh y tế ($1024 \times 1024$):
- Số lượng ứng viên trong Candidate Pool tăng mạnh, làm tăng thời gian tính toán của phép lọc không gian Spatial NMS và toán tử GCT.
- Mặc dù Directional Headroom Gain giúp thu hẹp không gian tìm kiếm, độ phức tạp của việc đánh giá hoán đổi liên minh bậc 2 ($O(M^2)$) sẽ đòi hỏi các kỹ thuật phân cấp thô-đến-tinh (coarse-to-fine hierarchy) trong tương lai.

### 1.2 Bản chất Heuristic của việc thoát cực tiểu cục bộ (Greedy Support Optima)
Việc tìm kiếm tập hỗ trợ $S \subseteq \{1, \dots, H \times W\}$ với $|S| \le K$ tối ưu toàn cục là bài toán NP-hard. Các toán tử của CASA:
- Hoán đổi 1-out / 1-in và 2-out / 2-in là các giải thuật tìm kiếm cục bộ (local search heuristics).
- Không tồn tại bảo đảm toán học (theoretical guarantee) rằng CASA sẽ luôn tìm được tập hỗ trợ tối ưu toàn cục nếu hàm mất mát bị kẹt trong một thung lũng bão hòa sâu.

### 1.3 Hiện tượng bão hòa biên hộp (Boundary Saturation)
Toán tử tính tiềm năng định hướng:
$$A_i(x) = \sum_c \left[ \max(g_c, 0)(1 - x_c) + \max(-g_c, 0)x_c \right]$$
hoạt động dựa trên giả định giá trị pixel chưa chạm biên. Đối với các vùng ảnh đồng nhất có độ tương phản cao (ảnh đen trắng hoặc bão hòa màu tuyệt đối), nhiều pixel tiềm năng có $A_i(x) \approx 0$ do không còn dư địa di chuyển theo hướng gradient, khiến thuật toán có thể bỏ sót các hướng tấn công phi tuyến tính bậc hai.

### 1.4 Giả định thiết kế ở vùng cực thưa (Low-$K$ Assumptions)
Toán tử GCT được thiết kế đặc thù cho $K=1$ và $K=2$. Thuật toán ngầm định rằng ở mức $K=1, 2$, không gian ứng viên $M=64$ đủ nhỏ để duyệt toàn diện. Khi $K \ge 4$, GCT tự động nhường chỗ cho Spatial NMS, đồng nghĩa với việc CASA chấp nhận từ bỏ việc duyệt chính xác từng cặp để kiểm soát chi phí tính toán.

---

## 2. Giới Hạn Thực Nghiệm (Experimental Limitations)

1. **Phạm vi tập dữ liệu:**  
   Các tuyên bố thực nghiệm chính hiện tại được chứng minh trên CIFAR-10 và CIFAR-100; kết quả Tiny-ImageNet đang trong giai đoạn hoàn thiện. Dự án chưa đánh giá đầy đủ trên toàn bộ 1.000 lớp của Full ImageNet-1k do giới hạn về ngân sách GPU.
2. **Kiến trúc mạng khảo sát:**  
   Nghiên cứu tập trung vào các kiến trúc mạng nơ-ron tích chập (Convolutional Neural Networks: ResNet-18, WideResNet-28-10, ResNet-50). Các kiến trúc dựa trên cơ chế Self-Attention như Vision Transformers (ViT, DeiT) hay Swin Transformer chưa được kiểm chuẩn trong giai đoạn này.
3. **Mô hình tấn công không chủ đích (Untargeted Focus):**  
   Toàn bộ các phân tích thực nghiệm đều tập trung vào bài toán tấn công không chủ đích (Untargeted Attack: ép mô hình dự đoán sai bất kỳ). Tấn công có chủ đích (Targeted Attack: ép mô hình dự đoán đúng một lớp mục tiêu $y_{\text{target}}$) có động lực tối ưu khác biệt và chưa được đưa vào phạm vi bài báo này.
4. **Không gian kỹ thuật số (Digital Space vs Physical Attack):**  
   CASA can thiệp ở mức độ pixel độc lập trong không gian số. Do đặc tính phân tán của các điểm ảnh thưa, nhiễu của CASA dễ bị suy thoái dưới các biến đổi vật lý (chụp ảnh qua camera, thay đổi góc nhìn, ánh sáng môi trường) và không được thiết kế cho kịch bản tấn công vật lý thế giới thực (physical adversarial patches).

---

## 3. Giới Hạn Của Nhánh Phòng Thủ (Defense Limitations)

1. **Chi phí huấn luyện đối nghịch thưa (CASA-AT Overhead):**  
   Tấn công thưa thớt bên trong vòng lặp huấn luyện min-max đòi hỏi nhiều bước tính gradient xuôi-ngược, khiến thời gian huấn luyện tăng gấp 15–20 lần so với huấn luyện chuẩn.
2. **Sự suy giảm độ chính xác sạch (Clean-Robust Trade-off):**  
   Mô hình phòng thủ thưa thớt (Sparse-PGD-AT) bị sụt giảm độ chính xác trên ảnh sạch từ $94.84\%$ xuống còn $87.20\%$, phản ánh sự đánh đổi không thể tránh khỏi giữa độ bền vững và năng lực biểu diễn của mạng.

---

## 4. Các Tuyên Bố Bị Nghiêm Cấm (Claims Explicitly NOT Supported)

Để duy trì chuẩn mực liêm chính học thuật, bài báo và các báo cáo liên quan **tuyệt đối không được đưa ra các tuyên bố sau**:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                 DANH MỤC CÁC TUYÊN BỐ KHÔNG ĐƯỢC PHÉP ĐƯA RA                │
├─────────────────────────────────────────────────────────────────────────────┤
│ [✗] CẤM: Tuyên bố "CASA tính toán chính xác giá trị Shapley / Marginal Gain"│
│     → Lý do: CASA chỉ dùng gradient-proxy A_i(x) để sàng lọc ứng viên, không│
│       thể tính toán toàn bộ 2^K tổ hợp trò chơi liên minh đầy đủ.           │
│                                                                             │
│ [✗] CẤM: Tuyên bố "CASA bảo đảm tìm được mức L_0 cực tiểu lý thuyết"        │
│     → Lý do: Cơ chế Drop-and-Repair là thuật toán nén heuristic, không phải │
│       bộ giải tối ưu toàn cục có chứng chỉ toán học (no certified minimum). │
│                                                                             │
│ [✗] CẤM: Tuyên bố "CASA là SOTA tuyệt đối trên mọi tác vụ thị giác máy tính"│
│     → Lý do: Thực nghiệm chỉ kiểm chứng trên phân loại ảnh 32×32 / 64×64;   │
│       chưa đánh giá phân vùng (Segmentation) hay phát hiện vật thể (YOLO).  │
│                                                                             │
│ [✗] CẤM: Tuyên bố "Các bộ lọc tiền xử lý tạo độ an toàn trước tấn công thưa"│
│     → Lý do: Đã bị bác bỏ bởi thực nghiệm BPDA (Adaptive Gap > +30%).       │
└─────────────────────────────────────────────────────────────────────────────┘
```
