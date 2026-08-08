Mình đã đọc trực tiếp repo `nxc1802/AA_2`, gồm benchmark runner, core metrics/projection, dataset/model pipeline, các attack custom, official adapters, defense, tests, proposed methods và cả các kết quả đang commit trong `result_marimo`.

## Kết luận tổng thể

**Repo có kiến trúc nghiên cứu khá tốt, nhưng hiện tại chưa đủ độ tin cậy khoa học để lấy các con số benchmark làm kết quả chính cho paper.**

Nếu chấm riêng:

| Hạng mục                                  |   Đánh giá |
| ----------------------------------------- | ---------: |
| Cấu trúc project                          |   **8/10** |
| Thiết kế benchmark A/B/C                  |   **8/10** |
| Metric pipeline                           | **7.5/10** |
| Reproducibility hiện tại                  |   **5/10** |
| Software engineering                      |   **6/10** |
| Độ faithful của baseline custom           |   **3/10** |
| Official baseline integration             |   **5/10** |
| Proposed-method implementation vs ý tưởng | **2.5/10** |
| Defense evaluation                        |   **4/10** |
| Mức sẵn sàng để viết paper                |  **~4/10** |

Điểm quan trọng nhất là: **framework tốt hơn algorithm implementation**. Nói cách khác, bộ khung của repo có thể giữ lại phần lớn, nhưng một số attack và toàn bộ 4 proposed method cần được xem xét lại trước khi chạy benchmark lớn.

---

# 1. Kiến trúc repo: đây là phần làm tốt

Repo đã được chia đúng kiểu một research codebase:

* `src/attacks/`
* `src/benchmark/`
* `src/core/`
* `src/datasets/`
* `src/models/`
* `src/defenses/`
* `src/reports/`
* `src/visualization/`
* `tests/`
* `third_party/`

Ngoài ra còn có registry attack, provenance của third-party, kết quả benchmark và notebook. Đây là hướng tổ chức tốt hơn rất nhiều so với việc nhét toàn bộ experiment vào notebook.

Mình đặc biệt đánh giá tốt cách repo tách attack thành:

**Group A — explicit K-budget**

→ chạy trực tiếp ở nhiều mức `K`.

**Group B — minimum-support**

→ attack một lần rồi đánh giá cumulative `ASR@K`.

**Group C — non-pixel-K**

→ FGSM/BIM/PGD/SFA, không ép vào cùng một khung K giả tạo.

Đây là taxonomy hợp lý về mặt experimental design.

---

# 2. Core (L_0) implementation là một trong những phần chắc nhất

Repo định nghĩa **spatial (L_0)** theo pixel thay vì từng scalar RGB: chỉ cần một channel tại pixel thay đổi thì pixel đó được tính là 1.

Đồng thời có:

* `exact_spatial_topk_mask`
* `project_l0`
* spatial score theo norm giữa các channels

Đây là lựa chọn đúng cho benchmark sparse attack trên ảnh. Test cũng kiểm tra projection thực sự thỏa `L0 <= K`.

Điểm này nên **giữ nguyên làm convention chính của paper**, sau đó ghi rõ:

[
|\delta|_{0,\text{pixel}}
=========================

\sum_{h,w}
\mathbf 1
\left[
\max_c|\delta_{c,h,w}|>\tau
\right].
]

Nó tránh được vấn đề một số repo khác tính RGB components, khiến cùng “(L_0=30)” nhưng ý nghĩa khác nhau.

---

# 3. Metric benchmark cũng có nền tảng tốt

Runner hiện tính:

* Clean Accuracy
* Robust Accuracy
* Accuracy Drop
* Conditional ASR
* spatial (L_0)
* (L_2)
* (L_\infty)
* PSNR
* SSIM
* LPIPS
* iterations
* runtime

Quan trọng hơn, `Conditional ASR` được tính trên những ảnh **clean model ban đầu phân loại đúng**:

[
ASR_{\text{cond}}
=================

\frac{
#{f(x)=y,\ f(x_{adv})\ne y}
}{
#{f(x)=y}
}.
]

Đây là cách đúng hơn việc lấy tất cả test samples làm mẫu số.

Phần distortion cũng có cả **all-sample** và **success-conditioned**, đây là thiết kế tốt. Với paper sparse attack, mình khuyên report distortion chính trên successful attacks, vì nếu attack fail và trả ảnh gần nguyên bản thì average distortion trên tất cả ảnh sẽ vô tình làm method trông “đẹp” hơn.

---

# 4. Vấn đề lớn nhất: nhiều baseline custom không phải implementation faithful của paper

Đây là vấn đề P0.

Repo hiện có hai lớp:

1. custom reimplementation;
2. official adapter.

Nhưng **default benchmark lại chạy custom reimplementation**, vì CLI mặc định `use_official=False`.

Điều đó rất nguy hiểm nếu bảng trong paper ghi đơn giản:

> Sparse-RS
> CornerSearch
> SparseFool
> Sigma-Zero
> BruSLe
> GSE

Reviewer mặc nhiên hiểu rằng đó là phương pháp tương ứng với paper gốc.

Trong repo hiện tại, nhiều cái thực chất chỉ là **inspired/proxy versions**.

---

## 4.1 Sparse-RS custom

Code hiện tại mỗi iteration:

* random K coordinates;
* random ±α;
* tạo candidate từ clean image;
* giữ candidate nếu CE loss tăng.

Đây đúng là một **random-search sparse attack**, nhưng quá đơn giản để gọi là Sparse-RS faithful.

Paper Sparse-RS là framework score-based query-efficient được thiết kế riêng cho (L_0), patch, frame và có cơ chế random-search có cấu trúc. Paper chính thức là AAAI 2022. ([AAAI][1])

### Đề xuất

Custom này đổi tên thành:

> `RandomSparseSearch` hoặc `SparseRS-inspired`

và **baseline Sparse-RS trong bảng paper phải chạy official implementation**.

---

# 5. CornerSearch custom vi phạm cả threat model

Đây là lỗi nghiêm trọng nhất ở Group A.

Custom CornerSearch hiện gọi:

```python
loss.backward()
grad_mag = x_adv.grad...
```

sau đó dùng gradient để chọn pixel.

Trong khi CornerSearch gốc là **black-box attack**. Paper ICCV 2019 nói rõ phương pháp hoạt động trong black-box scenario chỉ thông qua output classifier. ([Open Access CVF][2])

Vậy nếu bảng của bạn ghi:

| Method       | Threat model |
| ------------ | ------------ |
| CornerSearch | Black-box    |

nhưng code thực tế sử dụng gradient, thì kết quả **không hợp lệ**.

Custom method hiện tại về bản chất gần:

> greedy gradient-guided corner perturbation

hơn là CornerSearch.

### Cần làm

**Không dùng `CornerSearchAttack` custom để report baseline CornerSearch.**

Hoặc đổi tên rõ thành:

> `GradientCornerGreedy`

---

# 6. Official CornerSearch cũng đang bị làm yếu bất thường

Adapter official hiện tạo:

```python
'n_iter': min(self.max_iter, 5)
'n_max': min(max(1, self.k), 5)
```

Tức là dù:

[
K=32,64,128
]

thì một vài tham số quan trọng vẫn bị cap xuống **5**.

Trong paper CornerSearch, Croce & Hein dùng chẳng hạn `Niter=1000` trong benchmark gốc. ([Open Access CVF][3])

Vì vậy adapter hiện tại tuy gọi là `official-adapter` nhưng **không phải official configuration**.

Đây cần sửa trước benchmark.

---

# 7. SparseFool custom không phải SparseFool

Custom SparseFool hiện chủ yếu:

1. lấy margin gradient;
2. top-K gradient magnitude;
3. sign update;
4. lặp lại.

SparseFool gốc là một **geometry-inspired sparse attack**, dựa trên local linearization của decision boundary và projection lên boundary. CVPR 2019 mô tả rõ yếu tố hình học này. ([Open Access CVF][4])

Vì vậy:

> code hiện tại = DeepFool-inspired top-K gradient heuristic

chứ chưa phải SparseFool.

Một vấn đề nữa là support không được project toàn cục sau mỗi step. Nếu top-K thay đổi qua các iteration, tổng số pixel thay đổi có thể tăng rất nhiều.

---

# 8. Sigma-Zero custom sai bản chất khá rõ

File custom gọi:

> `SigmaZero Attack (ICLR 2025)`

nhưng thực tế algorithm là:

* EMA gradient;
* gradient magnitude;
* top-K;
* sparse sign update.

Trong khi σ-zero thật sử dụng:

* differentiable approximation của (L_0);
* gradient optimization;
* adaptive/dynamic thresholding;
* tìm adversarial example với minimum support.

Official repo cũng ghi rõ paper được accept **ICLR 2025**. ([GitHub][5])

Do đó custom `SigmaZeroAttack` hiện tại không nên mang tên Sigma-Zero trong bảng benchmark.

Điểm hay là repo thực sự đã vendor code Sigma-Zero và có `sigma_zero_attack.py`.

=> **hãy dùng code đó làm baseline chính**.

---

# 9. Có inconsistency metadata của Sigma-Zero

Trong các phần repo hiện có sự không thống nhất về venue/year của Sigma-Zero.

Adapter ghi:

> ICLR 2025

và official project cũng xác nhận ICLR 2025.  ([GitHub][5])

Registry cần thống nhất theo citation cuối cùng:

> Cinà et al., σ-zero, ICLR 2025.

Việc này nhỏ về code nhưng quan trọng cho paper bibliography.

---

# 10. Homotopy custom có một bug toán học đáng chú ý

Trong `HomotopyAttack`, repo tạo:

[
L =
L_{\mathrm{cls}}
----------------

\gamma\lambda
\sum_i
\frac{|\delta_i|}{|\delta_i|+\epsilon}.
]

Nghe có vẻ đúng hướng.

Nhưng `delta` tại chỗ đó không phải biến cần gradient mà update lại lấy:

```python
grad = x_adv.grad
```

Do đó sparsity proxy không thực sự đóng góp vào gradient đang dùng để update attack.

Nói cách khác, penalty Homotopy hiện tại gần như **không điều khiển optimization direction** như mục tiêu code muốn thể hiện. Sau đó method lại top-K gradient và `project_l0`.

Đây là bug cần sửa, không chỉ là khác biệt implementation.

---

# 11. GSE custom khác rất xa GSE thật

Repo implementation GSE hiện:

* gradient magnitude;
* average pooling 2×2;
* chọn top groups;
* sparse PGD;
* project (L_0).

GSE thật tại ICLR 2025 là một algorithm **hai phase**:

1. quasinorm optimization với (1/2)-quasinorm proximal operator;
2. projected Nesterov accelerated gradient với magnitude regularization.

([ML Anthology][6])

Vì vậy custom code hiện tại hợp lý nếu đặt tên:

> `BlockSparsePGD`

nhưng không nên report nó dưới tên `GSE`.

---

# 12. BruSLe custom hiện gần như hoàn toàn khác paper

Code:

* lấy `patch_dim = int(sqrt(K))`;
* random contiguous patch;
* random ±α;
* query model;
* accept nếu loss tốt hơn.

Trong khi BruSLeAttack là score-based **Bayesian sparse black-box algorithm**, ICLR 2024. ([ICLR Proceedings][7])

Ngoài vấn đề fidelity còn có một vấn đề fairness rất cụ thể.

Ví dụ:

[
K=8
]

thì:

```text
patch_dim = floor(sqrt(8)) = 2
```

=> thực tế chỉ thay:

[
2\times2=4
]

pixels.

Tương tự:

| Requested K | BruSLe thực dùng tối đa |
| ----------: | ----------------------: |
|           1 |                       1 |
|           2 |                       1 |
|           4 |                       4 |
|           8 |                       4 |
|          16 |                      16 |
|          32 |                      25 |
|          64 |                      64 |
|         128 |                     121 |

Nó vẫn thỏa (L_0\le K), nhưng **không tận dụng cùng budget**, nên so ASR trực tiếp không fair.

---

# 13. SAIF custom cũng không khớp SAIF thật

Implementation hiện tại gần như:

[
S_t=S_{t-1}+|\nabla_xL|,
]

chọn top-K cumulative gradient rồi PGD update.

Trong khi SAIF paper sử dụng **Frank-Wolfe / conditional gradient** để đồng thời kiểm soát magnitude và sparsity. ([arXiv][8])

Vậy SAIF custom cũng nên coi là proxy, không phải paper baseline.

---

# 14. Sparse-PGD cần dùng official implementation

Custom Sparse-PGD hiện:

* dense perturbation `p`;
* cumulative gradient magnitude `m_logits`;
* hard top-K mask;
* `delta = p * mask`.

Cấu trúc này hợp lý như một sparse-PGD heuristic, nhưng không đủ để khẳng định reproduce Sparse-PGD paper.

Paper chính thức sparse-PGD được ICML 2024 công bố với code `CityU-MLO/sPGD`. ([Proceedings of Machine Learning Research][9])

Đối với baseline SOTA này, nên ưu tiên official.

---

# 15. OnePixel là một trong những custom baseline tốt hơn, nhưng có bug state

OnePixel có cấu trúc DE khá đúng:

* population;
* mutation;
* crossover;
* selection;
* coordinates + RGB.

Tuy nhiên có một bug tiềm năng.

Sau khi:

```python
pop[improved] = trials[improved]
outs[improved] = trial_outs[improved]
```

`cand_imgs` không phải lúc nào cũng được decode lại.

Ở iteration sau code có thể:

```python
succ_mask = outs.argmax(...) != y
best_img = cand_imgs[first_succ]
```

Tức là prediction `outs` có thể thuộc **population mới**, nhưng `cand_imgs` vẫn thuộc population cũ.

=> có trường hợp phát hiện success nhưng trả về image stale.

### Fix

Sau selection:

```python
cand_imgs = decode_population(pop)
```

hoặc track images đồng bộ với population.

---

# 16. JSMA hiện cũng là JSMA-inspired

Code JSMA hiện:

* lấy second-highest class làm target;
* optimize target logit − true logit;
* lấy absolute gradient;
* chọn một spatial pixel;
* cộng `theta` cho cả RGB.

Đây không phải đầy đủ Jacobian Saliency Map formulation cổ điển với các điều kiện saliency/pair selection.

Nó vẫn là baseline có ích, nhưng nên gọi:

> `JSMA-inspired batched approximation`

nếu không implement faithful version.

---

# 17. Group A — ý tưởng benchmark rất tốt

Mình đồng ý với K grid:

[
K\in{1,2,4,8,16,32,64,128}.
]

Với CIFAR-10 có:

[
32\times32=1024\text{ spatial pixels}
]

thì tương ứng:

|   K | % pixels |
| --: | -------: |
|   1 |   0.098% |
|   2 |   0.195% |
|   4 |   0.391% |
|   8 |   0.781% |
|  16 |    1.56% |
|  32 |    3.13% |
|  64 |    6.25% |
| 128 |    12.5% |

Grid logarithmic như vậy tốt hơn `[1,2,3,...]` vì cho thấy toàn bộ sparsity–success curve.

Phần này mình sẽ **giữ**.

Nhưng chỉ giữ sau khi baseline implementation đã được chuẩn hóa.

---

# 18. Group B — formulation ASR@K đúng về ý tưởng nhưng chưa đúng ở implementation

Runner làm:

[
Success@K
=========

{attack\ succeeds}
\cap
{L_0(x_{adv}-x)\le K}.
]

Về lý thuyết, với một attack thật sự tìm **minimum support**, đây chính là cách rất đẹp để tạo cumulative ASR:

[
ASR@K
=====

P(K_{\min}\le K\mid f(x)=y).
]

### Nhưng vấn đề

Các attack trong Group B không đồng nhất về việc thật sự tìm (K_{\min}).

Ví dụ runner hiện gọi roughly:

* SparseFool với `k=250`;
* SigmaZero với cấu hình riêng;
* Homotopy target sparsity 250;
* GSE max groups;
* Pixle 20 swaps;
* FMSA budget 250.

Sau đó lấy (L_0) output và post-filter.

Đây **chỉ hợp lệ nếu output cuối thực sự đại diện cho solution support gần tối thiểu**.

Với custom SigmaZero/Homotopy/GSE/FMSA hiện tại, điều kiện đó không thỏa.

### Kết luận

**Giữ concept cumulative ASR@K, nhưng chỉ áp dụng cho method có semantics minimum-support.**

Đây thực ra là một trong những phần experimental design đáng giữ nhất của repo.

---

# 19. Có một mismatch (L_0) rất tinh vi với Sigma-Zero official

Benchmark của repo định nghĩa **spatial pixel (L_0)**.

Nhưng official Sigma-Zero code tính:

```python
true_l0 = active_delta.flatten(1).ne(0).sum(...)
```

tức là đếm **RGB scalar components**, không phải spatial pixels.

Nếu adapter truyền:

```python
epsilon_budget=K
```

thì early stopping bên Sigma-Zero hiểu:

[
K = \text{changed channel components}
]

trong khi benchmark hiểu:

[
K = \text{changed spatial pixels}.
]

Hai budget không tương đương.

### Nên làm

Không truyền `epsilon_budget=spatial_K` trực tiếp.

Với Group B minimum-support:

* để Sigma-Zero chạy minimum-support tự nhiên;
* sau đó **recompute spatial (L_0)** bằng metric chung của repo;
* dùng giá trị đó cho cumulative ASR@K.

Như vậy tất cả method được quy về một định nghĩa duy nhất.

---

# 20. Official adapter hiện chưa hoàn toàn đáng tin cậy về provenance

`SigmaZeroOfficialAdapter` có pattern:

```python
try:
    import official
except ImportError:
    define fallback implementation
```

Vấn đề là benchmark bên ngoài vẫn label:

> `official-adapter`

ngay cả khi ImportError khiến nó chạy fallback.

Vậy một dependency thiếu cũng có thể làm bảng CSV nói:

> Implementation Source = official-adapter

nhưng thực tế **không chạy official code**.

### Đây phải fail hard

Nên:

```text
official requested + import fails
          ↓
raise RuntimeError
```

Không fallback im lặng.

Nếu muốn fallback thì provenance phải trở thành:

```text
fallback-reimplementation
```

---

# 21. THIRD_PARTY.md cần sửa trước publication

Ý tưởng ghi provenance là rất tốt.

Nhưng hiện tài liệu vẫn chứa các entry kiểu:

```text
https://github.com/
```

cho SigmaZero, Sparse-PGD, Homotopy, GSE.

Điều này không đủ audit.

Mỗi external baseline nên ghi:

```text
paper
official repo URL
commit SHA
license
local vendored commit
adapter file
modifications
upstream config
```

Đặc biệt repo hiện không có license rõ ràng ở root trong cấu trúc mình đọc được, trong khi vendor nhiều third-party source. Đây nên được xử lý trước khi public research artifact.

---

# 22. Proposed methods hiện là phần cần làm lại nhiều nhất

Đây là phần quan trọng nếu mục tiêu cuối là paper.

Docs mô tả bốn ý tưởng khá khác biệt:

### CPA

> pixel cooperative interaction.

### FCSA

> coalition discovery,
> (\Delta F(S)-\sum_i\Delta F(i)).

### FMSA

> feature → minimal pixel support.

### HSA

> actual hypergraph giữa pixels và representations.

Nhưng code hiện tại chưa implement các formulation đó.

---

# 23. CPA hiện mới là local-smoothed saliency

CPA code:

[
g_i=|\nabla_iL|
]

sau đó:

[
s_i
===

g_i
+
\lambda
\operatorname{AvgPool}_{3\times3}(g)_i.
]

Rồi top-K + gradient sign.

Đây là:

> spatially smoothed / neighborhood-aware saliency

chứ chưa phải **cooperative pixel interaction**.

Nó chưa đo:

[
I(i,j)
]

hay:

[
F(S)-\sum_{i\in S}F(i).
]

### Nhận xét

CPA là **prototype tốt**, nhưng claim novelty cần hạ xuống hoặc algorithm phải nâng lên.

---

# 24. FCSA mismatch còn lớn hơn

Docs định nghĩa coalition score dựa trên joint effect.

Nhưng code hiện tính mỗi pixel độc lập:

[
score_i
=======

mean_c(|g_{ic}|)
\times
max_c(|g_{ic}|).
]

Sau đó top-K.

Không có:

* candidate coalition (S);
* joint perturbation evaluation;
* interaction;
* synergy;
* marginal contribution;
* feature collapse;
* coalition search.

Do đó hiện tại FCSA **không phải coalition attack**.

Nếu submit paper với formulation trong docs và code này, reviewer có thể bắt được rất nhanh.

---

# 25. FMSA hiện là feature-loss sparse PGD, không phải minimal-support search

FMSA làm được một thứ thực sự có giá trị:

* hook `layer4`;
* lấy clean representation;
* attack CE + feature difference;
* top-K gradient;
* project support.

Đây có thể được diễn giải chính xác là:

> **Feature-disruption-guided sparse attack**

Nhưng chưa phải:

> Feature-to-Minimal-Support Attack.

Không có explicit search:

[
\min_{\delta}|\delta|_0
\quad
s.t.
\quad
D(F(x+\delta),F(x))\ge\tau.
]

Đặc biệt runner gọi cùng class với `support_budget=250` rồi đặt tên:

> `FMSA-minimal-support`

điều này chưa có cơ sở algorithmic.

---

# 26. HSA hiện chưa có hypergraph

HSA code thực hiện:

[
s
=

g
+0.5AvgPool_3(g)
+0.3AvgPool_5(g)
+0.2AvgPool_7(g)
]

rồi top-K.

Đây là **multi-scale spatial saliency**.

Không có:

* incidence matrix (H);
* nodes;
* explicit hyperedges;
* node–hyperedge relationships;
* hyperedge weights;
* hypergraph Laplacian;
* centrality trên hypergraph thật;
* hyperedge disruption objective.

Vậy tên `HypergraphSparseAttack` hiện overclaim khá mạnh.

Nếu muốn giữ algorithm như hiện tại, tên phù hợp hơn là:

> `MultiScaleSaliencySparseAttack`.

Nếu muốn giữ paper idea **HSA**, cần viết lại phần algorithm.

---

# 27. Bảng tóm tắt độ faithful hiện tại

| Method              | Tình trạng code                             | Có nên dùng kết quả hiện tại cho paper? |
| ------------------- | ------------------------------------------- | --------------------------------------- |
| FGSM                | tương đối chuẩn                             | ✅                                       |
| BIM                 | tương đối chuẩn                             | ✅                                       |
| PGD                 | tương đối chuẩn                             | ✅                                       |
| JSMA                | simplified                                  | ⚠️                                      |
| OnePixel            | khá gần DE, có bug state                    | ⚠️                                      |
| CornerSearch custom | **sai threat model**                        | ❌                                       |
| SAIF custom         | khác paper                                  | ❌                                       |
| PGD0 custom         | heuristic                                   | ⚠️                                      |
| Sparse-PGD custom   | simplified                                  | ❌ nếu gọi official baseline             |
| Sparse-RS custom    | generic random search                       | ❌                                       |
| BruSLe custom       | khác paper rất nhiều                        | ❌                                       |
| SparseFool custom   | khác algorithm gốc                          | ❌                                       |
| Sigma-Zero custom   | khác algorithm gốc                          | ❌                                       |
| Homotopy custom     | có bug objective                            | ❌                                       |
| GSE custom          | khác GSE thật                               | ❌                                       |
| Pixle custom        | random swap approximation                   | ⚠️                                      |
| CPA                 | prototype                                   | ⚠️                                      |
| FCSA                | chưa có coalition                           | ❌ cho claim hiện tại                    |
| FMSA                | feature-guided attack, chưa minimal support | ⚠️                                      |
| HSA                 | chưa phải hypergraph                        | ❌ cho claim hiện tại                    |

---

# 28. PGD0 custom có một vấn đề optimization thú vị

`PGD0Attack` làm:

[
\delta_{t+1}
============

\Pi_{L_0\le K}
(\delta_t+\alpha,sign(g)).
]

Ở iteration đầu:

[
\delta_0=0.
]

Với sign-gradient, gần như tất cả nonzero pixels nhận magnitude bằng (\alpha), nên khi `project_l0` rank theo magnitude, rất nhiều pixel bị tie.

Kết quả: top-K ban đầu có thể phụ thuộc chủ yếu vào behavior tie-breaking, chứ không phải pixel nào có gradient lớn hơn.

Sau đó các pixel đã chọn tích lũy magnitude và support dễ bị “lock”.

### Tốt hơn

Dùng một trong:

* score = gradient magnitude cho support selection;
* magnitude update riêng;
* random restart;
* support swapping;
* official PGD-(L_0).

---

# 29. Sampling benchmark có một bug reproducibility

Code comment:

> `Fixed Stratified / Deterministic Sample Indexing`

nhưng thực tế dùng:

```python
torch.randperm(...)
```

không có stratification.

Vậy phải sửa một trong hai:

* đổi comment thành deterministic random sampling;
* hoặc implement real class-stratified sampling.

Mình nghiêng về **real stratification**, vì benchmark 1,000 CIFAR images rất dễ lấy đúng 100 ảnh/class.

---

# 30. File benchmark indices có bug khi thay `num_samples`

Runner dùng một file:

```text
benchmark_indices_seed42.json
```

sau đó nếu file tồn tại:

```python
sampled_indices = json.load(f)[:n_eval]
```

Giả sử:

1. lần đầu chạy 10 samples → lưu 10 indices;
2. lần sau yêu cầu 1000;
3. file đã tồn tại;
4. slice `[:1000]` vẫn chỉ có 10.

Nhưng metadata lại dùng requested `n_eval`.

=> có thể request **1000 nhưng thực tế evaluate 10**.

### Fix

File nên chứa full deterministic permutation 10,000 indices, hoặc filename gồm sample count:

```text
cifar10_test_indices_seed42_n1000.json
```

và luôn assert:

```python
assert len(sampled_indices) == n_eval
```

---

# 31. Metadata experiment vẫn còn thiếu những thứ quan trọng

Hiện metadata có:

* timestamp;
* seed;
* num samples;
* device;
* CUDA name;
* indices hash;
* official/custom flag.

Đó là một khởi đầu tốt.

Nhưng paper-grade artifact nên thêm:

```text
git commit SHA
dataset name
dataset revision/hash
checkpoint SHA256
checkpoint source/version
model architecture
clean accuracy full test
attack config dump
K values
PyTorch version
CUDA version
Python version
LPIPS version
metric implementation version
hostname/GPU
```

Quan trọng nhất là **checkpoint hash**.

---

# 32. Có thể benchmark model random mà không báo lỗi

Ở `__main__`:

1. tạo ResNet18 random;
2. tìm checkpoint;
3. nếu tìm thấy thì load;
4. nếu không tìm thấy → tiếp tục benchmark model random.

Đây là P0 engineering bug.

Một benchmark adversarial trên untrained model hoàn toàn không có ý nghĩa.

### Nên đổi thành

```python
ckpt = find_existing_checkpoint(...)
if ckpt is None:
    raise FileNotFoundError(...)
```

Chỉ cho phép random model khi có flag explicit như:

```text
--allow-untrained-model
```

---

# 33. Dataset split khá tốt

Dataset loader làm:

[
50,000
\rightarrow
40,000\ train
+
10,000\ val
]

bằng stratified split, seed cố định.

Test 10,000 được giữ riêng.

Đây đúng với protocol mà mình nghĩ phù hợp cho project này.

Không cần chia lại nữa.

---

# 34. Nhưng dataset từ HF cần pin revision

Hiện load:

```python
load_dataset("Cuong2004/AA", ...)
```

không pin commit/revision.

Nếu HF dataset thay đổi sau này:

> cùng repo commit + cùng seed ≠ cùng dataset.

Nên pin:

```text
dataset_repo_revision
```

hoặc lưu SHA/hash của test samples.

---

# 35. Training model có vài điểm nên chỉnh

ResNet18 được adapt khá chuẩn cho CIFAR:

* conv 3×3 stride 1;
* bỏ maxpool;
* classifier 10 classes.

Optimizer:

* SGD;
* lr 0.1;
* momentum 0.9;
* WD (5\times10^{-4});
* cosine schedule.

Đây là recipe ổn.

Nhưng validation hiện chỉ check khoảng mỗi 20 epochs và best checkpoint chỉ được chọn từ những điểm đó.

Không cần validation quá thường xuyên, nhưng mình sẽ dùng:

> validate mỗi 5 epochs

hoặc mỗi epoch vì CIFAR validation rất rẻ.

---

# 36. Training reproducibility chưa đầy đủ

`set_seed()` tồn tại và khá đầy đủ:

* Python;
* NumPy;
* PyTorch;
* CUDA;
* deterministic cuDNN.

Nhưng `train_clean_resnet18()` bản thân không gọi seed trước initialization/training.

Vì vậy reproducibility phụ thuộc vào caller.

### Nên để

```python
def train_clean_resnet18(..., seed=42):
    set_seed(seed)
```

ngay trong training entrypoint.

---

# 37. Batch augmentation cần kiểm tra

Training cache toàn bộ train set lên VRAM rồi chạy:

```python
bx = train_aug(X_shuffled[i:i+batch_size])
```

Cần verify rằng phiên bản torchvision đang dùng tạo **random crop/flip độc lập cho từng ảnh**, chứ không áp cùng random parameters cho toàn batch tensor.

Nếu cùng crop/flip cho batch, augmentation diversity giảm đáng kể.

Không nhất thiết chắc chắn lỗi trên mọi torchvision version, nhưng nên có unit test đơn giản cho điểm này.

---

# 38. `requirements.txt` không đủ để reproduce

Hiện tất cả dependency gần như:

```text
torch>=2.0
numpy>=...
torchvision>=...
```

Với ML research, đây là quá lỏng.

Một người chạy tháng 8/2026 và người chạy năm sau có thể nhận:

* torch khác;
* torchvision khác;
* datasets khác;
* LPIPS khác;
* scipy khác.

### Nên có

```text
requirements-lock.txt
```

hoặc `uv.lock` / `conda-lock`.

---

# 39. Runtime hiện không thực sự là attack runtime

Runner bắt timer trước toàn bộ evaluation loop.

Trong loop lại có:

* clean prediction;
* attack;
* adversarial prediction;
* (L_0,L_2,L_\infty);
* PSNR;
* SSIM;
* LPIPS.

Rồi mới dừng timer.

Do đó:

> `Time/Img`

không phải pure attack generation time.

Nó là:

[
T_{\rm attack}
+
T_{\rm eval}
+
T_{\rm metrics}
+
T_{\rm LPIPS}.
]

Đối với paper thì sai semantics.

### Tách thành

```text
Attack Time / Image
Metric Time / Image
End-to-End Time / Image
```

Với GPU nên dùng CUDA events hoặc synchronize chỉ quanh attack call.

---

# 40. Query count hiện còn thiếu

Đặc biệt với:

* OnePixel
* CornerSearch
* Sparse-RS
* BruSLe
* Pixle

**query count là metric chính**, đôi khi còn quan trọng hơn runtime.

THIRD_PARTY.md cũng tự ghi rằng query/gradient evaluations cần report.

Nhưng current main attack table chủ yếu dùng `Avg Iterations`.

Iteration không tương đương query.

Ví dụ OnePixel:

[
1 \text{ iteration}
]

có thể evaluate cả population 20 samples.

Nên có API thống nhất:

```python
attacker.last_queries
attacker.last_grad_evals
attacker.last_forward_evals
```

---

# 41. Defense benchmark hiện chưa đủ chuẩn adversarial robustness

Hiện defense benchmark chỉ gồm preprocessing:

* Gaussian blur
* Median filter
* JPEG
* TVM

và test trên:

* PGD
* JSMA
* PGD0.

Nó báo:

[
Recovery
========

## Acc_{defended}

Acc_{undefended}.
]

Có ba thiếu sót lớn.

### A. Không report clean utility

Ví dụ blur có thể tăng attacked accuracy nhưng làm giảm clean accuracy rất nhiều.

Phải có:

[
CleanAcc_{defense}.
]

### B. Non-adaptive evaluation

Attack hiện craft:

[
x_{adv}
=======

A(f,x)
]

rồi mới:

[
f(D(x_{adv})).
]

Một adaptive attacker phải attack:

[
f\circ D.
]

Nếu preprocessing không differentiable thì dùng BPDA/EOT khi phù hợp.

Nếu không, defense dễ nhìn mạnh chỉ vì gradient masking.

### C. Chưa có adversarial training

Nếu mục tiêu dự án là đánh giá cả preprocessing và adversarial training thì repo hiện mới có preprocessing.

---

# 42. Tests hiện chủ yếu kiểm tra “chạy được”, chưa kiểm tra “đúng paper”

Tests hiện kiểm tra khá tốt các contract:

* shape;
* pixel range;
* (L_0\le K);
* projection;
* metric output;
* defense execution.

Official adapter tests cũng chủ yếu kiểm tra:

* chạy được;
* shape;
* một vài L0 constraint.

Nhưng chưa có regression test cho scientific equivalence.

Mình sẽ bổ sung tối thiểu:

1. exact output comparison custom vs official trên deterministic tiny case;
2. deterministic attack dưới same seed;
3. ASR non-decreasing theo K cho Group A, trong tolerance;
4. `max L0 <= K`;
5. conditional ASR formula;
6. Group B cumulative monotonicity;
7. official-adapter must really import official source;
8. no checkpoint → benchmark must fail;
9. requested 1000 samples → exactly 1000 evaluated;
10. query counter correctness;
11. clean/adv batch-size invariance;
12. SSIM/PSNR cross-validation với reference library.

---

# 43. Kết quả đang commit hiện chưa thể được coi là benchmark

File hiện tại cho các ASR tăng theo:

```text
0%, 10%, 20%, 30%, ...
```

rất rõ rệt.

`full_attack_metrics.csv` cũng cho Clean Accuracy = 100% ở toàn bộ sample và các mức thay đổi theo bước 10 percentage points.

Điều này phù hợp với việc đây là **tiny smoke test khoảng 10 clean-correct images**, không phải benchmark có statistical meaning.

Do đó các số kiểu:

> Sparse-RS 100%
> Sparse-PGD 100%
> CPA 90%

trong artifact hiện tại **tuyệt đối chưa nên dùng để so sánh method**.

Chúng chỉ cho biết:

> pipeline chạy được.

---

# 44. Còn có mismatch giữa kết quả commit và code benchmark hiện tại

CSV đang commit có schema như:

```text
ASR (%)
Avg L0
Avg L0 Ratio
Max L0
```

Trong khi runner hiện tại xuất những field như:

```text
Conditional ASR (%)
All Avg L0
Success Avg L0
Success Median L0
...
```

Điều này cho thấy artifact `result_marimo` hiện tại không được tạo hoàn toàn bởi phiên bản benchmark source đang ở HEAD.

Đây là một reproducibility issue.

### Nên để mỗi run có

```text
results/
└── <run_id>/
    ├── config.yaml
    ├── metadata.json
    ├── git_commit.txt
    ├── metrics.csv
    ├── per_sample.csv
    └── figures/
```

Không overwrite kết quả giữa các version.

---

# 45. Checkpoint hiện thực sự được download từ HF

Log commit cho thấy các run gần đây tải:

```text
Cuong2004/AA/models/resnet18_cifar10_best.pth
```

và có snapshot hash cụ thể trên HF cache.

Đó là điểm tốt.

Nhưng snapshot/hash đó cần đưa luôn vào benchmark metadata thay vì chỉ nằm trong log.

---

# 46. Thứ tự mình khuyên sửa

Nếu mục tiêu là paper, **đừng vội chạy 1000–10000 ảnh ngay**.

Thứ tự tốt nhất là:

| Priority | Việc                                           |
| -------- | ---------------------------------------------- |
| **P0**   | Freeze definition/threat model từng attack     |
| **P0**   | Official baseline phải chạy official thật      |
| **P0**   | Xóa/đổi tên custom proxies gây nhầm lẫn        |
| **P0**   | Sửa CornerSearch threat-model violation        |
| **P0**   | Sửa Homotopy objective                         |
| **P0**   | Sửa OnePixel stale state                       |
| **P0**   | Sửa sample-index bug                           |
| **P0**   | Fail nếu checkpoint không tồn tại              |
| **P0**   | Viết lại proposed method theo đúng formulation |
| **P1**   | Chuẩn hóa query/gradient counters              |
| **P1**   | Tách attack runtime khỏi metric runtime        |
| **P1**   | Pin dependency + dataset/checkpoint hashes     |
| **P1**   | Per-sample results                             |
| **P1**   | Add scientific regression tests                |
| **P1**   | Adaptive defense evaluation                    |
| **P2**   | CLI/config YAML/Hydra                          |
| **P2**   | CI                                             |
| **P2**   | polish docs/report                             |

---

# 47. Kiến trúc benchmark mình sẽ giữ sau khi refactor

Sau khi chỉnh, mình vẫn giữ thiết kế cốt lõi hiện tại:

### Group A — Fixed-budget attacks

Chạy:

[
K={1,2,4,8,16,32,64,128}.
]

Report:

[
ASR(K),\quad RA(K),\quad
L_0,\ L_\infty,\ PSNR,\ SSIM,\ LPIPS.
]

Vẽ:

[
ASR\text{ vs }K.
]

---

### Group B — Minimal-support attacks

Mỗi sample trả:

[
K_i^\star.
]

Sau đó:

[
ASR@K
=====

\frac{
\sum_i\mathbf 1[K_i^\star\le K]
}{
N_{\rm clean-correct}
}.
]

Report thêm:

[
Median(K^\star),\quad Mean(K^\star).
]

Đây mới là cách cực kỳ đẹp để so Sigma-Zero, SparseFool và các genuine minimum-support approaches.

---

### Group C — Dense/non-spatial-budget

FGSM/BIM/PGD/SFA được report riêng theo:

* (\epsilon);
* ASR;
* Robust Accuracy;
* (L_\infty);
* PSNR/SSIM/LPIPS.

Không cố nhét vào ASR@K.

---

# 48. Proposed method nên chọn **một**, không giữ cả bốn dưới dạng “ours”

Đây là điều mình sẽ thay mạnh nhất.

Hiện repo đưa:

* CPA
* FCSA
* FMSA
* HSA

đều vào benchmark như `ours`.

Nhưng bốn cái thực chất là **bốn research directions**, chưa nên coi là bốn proposed algorithms cùng lúc.

Cách hợp lý hơn:

```text
CPA prototype
FCSA prototype
FMSA prototype
HSA prototype
        ↓
pilot experiment 100–500 images
        ↓
chọn 1 method mạnh nhất
        ↓
formalize
        ↓
ablation
        ↓
full benchmark
```

Trong trạng thái code hiện tại, nếu phải chọn một nền tảng dễ nâng cấp nhất, **FMSA có implementation substance tốt nhất** vì nó đã thực sự đưa feature representation vào objective. CPA đứng thứ hai vì implementation đơn giản và rõ ràng.

FCSA/HSA hiện khoảng cách giữa claim và implementation quá lớn.

---

# 49. Repo hiện nên được định vị như thế nào?

Mình sẽ gọi trạng thái hiện tại là:

> **research benchmark scaffold + attack prototypes**

chứ chưa phải:

> **validated sparse adversarial attack benchmark**.

Đó không phải vấn đề quá tệ — thực ra phần khó của việc tổ chức framework đã làm khá tốt.

Điều cần tránh là lấy output hiện tại rồi bắt đầu viết Table 1/Table 2 của paper.

---

# 50. Đánh giá cuối cùng

### Những gì nên giữ

**Giữ gần như nguyên:**

* cấu trúc module;
* Group A/B/C;
* spatial (L_0);
* K grid;
* conditional ASR;
* success-conditioned distortion;
* 40k/10k train/val;
* held-out CIFAR test;
* attack registry concept;
* official adapter concept;
* metadata/result infrastructure.

### Những gì phải sửa trước khi full benchmark

**Phải sửa:**

* fidelity của baselines;
* provenance official implementation;
* CornerSearch custom;
* SparseFool/SigmaZero/GSE/BruSLe/SAIF custom naming;
* Group B semantics;
* sample-index bug;
* no-checkpoint behavior;
* query count;
* runtime methodology;
* defense methodology.

### Những gì cần thiết kế lại cho contribution

**Thiết kế lại đáng kể:**

* FCSA;
* HSA;
* FMSA minimal-support formulation;
* CPA interaction/cooperation definition.

---

## Phán đoán ngắn nhất

**Repo hiện tại có một benchmark framework tốt, nhưng algorithm layer chưa đáng tin cậy tương ứng.**

Nếu sửa đúng thứ tự, mình nghĩ không cần rewrite project. Có thể giữ khoảng **70–80% infrastructure**, nhưng trước khi chạy full experiment nên **thay/chuẩn hóa phần baseline attack và chọn lại một proposed method duy nhất**.

Điểm cần xử lý đầu tiên không phải hyperparameter hay tăng sample size, mà là **“method identity”: code đang chạy có thực sự là phương pháp mà tên của nó tuyên bố hay không**. Với trạng thái hiện tại, câu trả lời cho khá nhiều baseline và 3/4 proposed ideas là **chưa**.

[1]: https://ojs.aaai.org/index.php/AAAI/article/view/20595?utm_source=chatgpt.com "Sparse-RS: A Versatile Framework for Query-Efficient Sparse Black-Box Adversarial Attacks | Proceedings of the AAAI Conference on Artificial Intelligence"
[2]: https://openaccess.thecvf.com/content_ICCV_2019/html/Croce_Sparse_and_Imperceivable_Adversarial_Attacks_ICCV_2019_paper.html?utm_source=chatgpt.com "ICCV 2019 Open Access Repository"
[3]: https://openaccess.thecvf.com/content_ICCV_2019/papers/Croce_Sparse_and_Imperceivable_Adversarial_Attacks_ICCV_2019_paper.pdf?utm_source=chatgpt.com "Sparse and Imperceivable Adversarial Attacks"
[4]: https://openaccess.thecvf.com/content_CVPR_2019/html/Modas_SparseFool_A_Few_Pixels_Make_a_Big_Difference_CVPR_2019_paper.html?utm_source=chatgpt.com "CVPR 2019 Open Access Repository"
[5]: https://github.com/sigma0-advx/sigma-zero/blob/main/readme.md?utm_source=chatgpt.com "sigma-zero/readme.md at main · sigma0-advx/sigma-zero · GitHub"
[6]: https://mlanthology.org/iclr/2025/sadiku2025iclr-gse/?utm_source=chatgpt.com "GSE: Group-Wise Sparse and Explainable Adversarial Attacks | ML Anthology"
[7]: https://proceedings.iclr.cc/paper_files/paper/2024/hash/efb7f7530e6630dc7dbbfb783207898a-Abstract-Conference.html?utm_source=chatgpt.com "BRUSLEATTACK: A QUERY-EFFICIENT SCORE- BASED BLACK-BOX SPARSE ADVERSARIAL ATTACK"
[8]: https://arxiv.org/abs/2212.07495?utm_source=chatgpt.com "SAIF: Sparse Adversarial and Imperceptible Attack Framework"
[9]: https://proceedings.mlr.press/v235/zhong24c.html?utm_source=chatgpt.com "Towards Efficient Training and Evaluation of Robust Models against $l_0$ Bounded Adversarial Perturbations"



Tôi thắc mắc về custom và official. Các method official là lấy code trực tiếp từ paper còn custom là implement lại theo mô tả của paper đúng chứ? Vậy thì method nào có official rồi thì đâu cần custom nữa? Thậm chí là các method không có official mà chỉ có thể custom cũng nên xem xét việc có nên tiếp tục đưa vào baseline/benchmark hay không
Đúng, cách hiểu của bạn về mặt nguyên tắc là như vậy:

* Official: dùng implementation do tác giả paper phát hành, hoặc code chính thức được paper/repo tác giả chỉ định.
* Custom: tự reimplement lại method dựa trên paper, công thức, pseudocode và mô tả experimental setup.

Và từ góc độ làm benchmark cho paper, mình đồng ý với hướng bạn nêu: nếu đã có official implementation đủ dùng, không nên giữ custom version như baseline chính, trừ khi có lý do rất cụ thể như official code không chạy được trên framework/model hiện tại và bạn cần một bản port có kiểm chứng equivalence.

Vấn đề của repo hiện tại là nó đang trộn hai mục đích khác nhau. Một mặt custom implementations có ích cho việc học thuật, debug, test API và chạy smoke test. Mặt khác khi chúng xuất hiện cạnh official implementation với cùng tên method, chúng tạo ra rủi ro rất lớn: người đọc dễ hiểu rằng cả hai đều đại diện cho cùng algorithm, trong khi thực tế custom có thể chỉ gần giống hoặc thậm chí khác đáng kể.

Mình sẽ đề xuất policy cho AA_2 như sau:

Trạng thái method	Đưa vào benchmark chính?	Cách xử lý
Có official code, chạy được	Có	Chỉ dùng official
Có official nhưng cần adapter	Có	Adapter mỏng, không sửa algorithm
Có official nhưng code hỏng/không tương thích	Có điều kiện	Port/reimplement, nhưng phải validate với official
Không có official, mô tả paper đủ chi tiết	Có điều kiện	Custom + ghi rõ reimplementation
Không có official, paper thiếu chi tiết quan trọng	Không nên	Loại khỏi benchmark chính
Chỉ tìm thấy implementation bên thứ ba	Có điều kiện mạnh	Chỉ dùng nếu được cộng đồng/paper khác tin cậy và audit được
Custom chỉ “inspired by” paper	Không	Đổi tên thành phương pháp riêng hoặc bỏ

Với các method có official, custom có thể vẫn tồn tại trong repo nhưng nên đưa ra khỏi experimental benchmark, chẳng hạn vào src/attacks/reimplementations/ hoặc legacy/, và không để runner chính import chúng. Điều này giúp codebase sạch hơn và quan trọng nhất là tránh vô tình chạy nhầm custom khi tạo bảng kết quả.

Điểm thứ hai bạn nói còn quan trọng hơn: không phải method nào không có official cũng đáng reimplement để benchmark. Việc đưa một custom baseline vào paper chỉ hợp lý khi chúng ta có khả năng tái hiện method đủ chính xác. Nếu paper thiếu pseudocode, thiếu hyperparameter, không rõ stopping criterion, normalization, target selection, projection, query budget… thì custom implementation rất dễ trở thành “một algorithm mới mang tên algorithm cũ”. Trong trường hợp đó, bỏ baseline đó còn tốt hơn đưa một kết quả không đáng tin.

Với AA_2, mình sẽ chia baseline thành ba tier:

1. Tier A — authoritative baseline: official author code hoặc implementation chính thức trong thư viện đáng tin cậy. Đây là các method được phép vào bảng benchmark chính.
2. Tier B — verified reimplementation: không có official usable code nhưng custom implementation đã được đối chiếu kỹ với paper, reproduction numbers và các invariants của algorithm. Có thể vào bảng, nhưng ghi rõ Reimplementation.
3. Tier C — heuristic/proxy: implementation chỉ dựa trên ý tưởng chung. Không được mang tên paper baseline trong bảng chính.

Theo tiêu chuẩn đó, nhiều custom hiện tại trong repo thực chất đang ở Tier C chứ chưa phải Tier B.

Ví dụ rất rõ là custom CornerSearch: paper là black-box nhưng implementation custom dùng gradient. Cái này không thể gọi là CornerSearch reimplementation nữa. Tương tự custom Sigma-Zero chỉ dùng EMA gradient + top-K, trong khi σ-zero thật có differentiable L_0 approximation và dynamic thresholding. Những cái đó nên loại khỏi baseline ngay, không cần cố sửa nếu official đã có.

Vì vậy, architecture mình khuyên cho repo sẽ là:

src/attacks/
├── official/
│   ├── sparse_rs.py
│   ├── corner_search.py
│   ├── sigma_zero.py
│   ├── sparsefool.py
│   ├── spgd.py
│   └── ...
│
├── reimplemented/
│   ├── fgsm.py
│   ├── bim.py
│   ├── pgd.py
│   └── <only methods without usable official code>
│
├── proposed/
│   └── ...
│
└── experimental/
    └── heuristic prototypes

Còn attack_registry.yaml nên ghi rất rõ:

implementation:
  type: official
  upstream_repo: ...
  upstream_commit: ...

hoặc:

implementation:
  type: reimplementation
  validation: reproduction_checked

Và benchmark runner chỉ chấp nhận official + verified reimplementation, không chạy experimental.

Đối với danh sách hiện tại của AA_2, mình sẽ làm bước tiếp theo là audit toàn bộ baseline và rút gọn danh sách: method nào có official thì giữ official và xóa khỏi benchmark custom; method nào không có official thì kiểm tra paper/code availability, rồi quyết định KEEP / REIMPLEMENT / DROP. Đây sẽ làm benchmark nhỏ hơn nhưng đáng tin hơn rất nhiều, và đó là lựa chọn đúng cho paper.