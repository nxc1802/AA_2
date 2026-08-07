Mình đã đọc cấu trúc repo, benchmark engine, model/dataset pipeline, metric, test, adapter third-party, các proposed attack và cả kết quả đang commit trong result_marimo.

Kết luận tổng quan

Repo AA_2 đã có nền tảng khá tốt cho một research codebase về Sparse Adversarial Attack, đặc biệt ở cách chia module và chuẩn hóa spatial L_0. Tuy nhiên, ở trạng thái hiện tại mình chưa khuyên dùng các bảng kết quả trong repo làm kết quả paper.

Vấn đề chính không phải thiếu attack. Ngược lại, repo đang có quá nhiều attack. Vấn đề chính là:

1. benchmark protocol chưa đủ chặt để so sánh paper-level;
2. metric distortion đang bị bias bởi failed attacks;
3. nhãn "Official" hiện chưa thực sự đồng nghĩa tất cả method dùng official implementation;
4. black-box chưa đo query count;
5. kết quả commit hiện tại thực chất là mức smoke test khoảng 10 ảnh;
6. defense pipeline hiện chưa runnable;
7. một số proposed method có khoảng cách khá lớn giữa tên/concept mô tả và thuật toán thực sự được code;
8. reproducibility chưa đủ: dependency, config, checkpoint revision, sample indices, multiple seeds.

Nếu chấm trạng thái hiện tại:

Thành phần	Đánh giá
Kiến trúc source	8/10
Phạm vi attack benchmark	9/10
Spatial L_0 implementation	8.5/10
Benchmark fairness	5/10
Reproducibility	4/10
Defense pipeline	2/10
Proposed-method maturity	4–6/10 tùy method
Paper-ready	~4.5/10

Điều đáng mừng là phần lớn điểm yếu nằm ở protocol và engineering, tức sửa được mà không cần viết lại toàn repo.

⸻

1. Kiến trúc repo: đây là điểm mạnh

Repo được tổ chức tương đối đúng kiểu research project:

src/
├── attacks/
│   ├── baselines/
│   ├── classical/
│   ├── optimization/
│   ├── attention_attribution/
│   ├── frequency/
│   ├── blackbox/
│   ├── proposed/
│   └── adapters/
├── benchmark/
├── core/
├── datasets/
├── models/
├── reports/
└── visualization/
tests/
third_party/
docs/
notebook/
result_marimo/

Danh mục attack hiện khá rộng: FGSM/BIM/PGD; JSMA, OnePixel, CornerSearch, SparseFool; PGD0, Sparse-PGD, Sigma-Zero, GSE, Homotopy, SAIF; Sparse-RS, BruSLe, Pixle; cùng CPA/FCSA/FMSA/HSA proposed.

Việc tách:

* implementation riêng;
* official adapter riêng;
* benchmark engine riêng;
* core metric/projection riêng;
* report/visualization riêng;

là hướng đúng.

Đặc biệt mình đánh giá cao việc không nhét toàn bộ experiment vào một notebook khổng lồ.

⸻

2. Phần L_0 hiện được làm khá đúng

Một trong những phần tốt nhất của repo là:

channel_max = delta.abs().max(dim=1).values
l0_per_sample = (channel_max > eps).flatten(1).sum(dim=1)

Tức là một pixel (h,w) được tính là modified nếu ít nhất một trong 3 channel thay đổi.

Như vậy trên CIFAR-10:

L_0 \in [0,1024]

chứ không phải [0,3072].

Đây là định nghĩa phù hợp cho pixel-sparse attack.

Ngoài ra:

exact_spatial_topk_mask(...)
project_l0(...)

đã giải quyết được vấn đề threshold tie khiến số pixel đôi lúc vượt quá K.

Một edge case nhỏ

exact_spatial_topk_mask() hiện có:

k_bounded = min(max(1, k), num_pixels)

Nghĩa là:

K=0 → K=1

Nên hàm chưa biểu diễn được perturbation budget bằng 0.

Không ảnh hưởng K sweep hiện tại vì bạn bắt đầu từ 1, nhưng nên sửa:

if k <= 0:
    return torch.zeros(..., dtype=torch.bool)

⸻

3. Cách chia Group A/B/C về cơ bản là hợp lý

Benchmark hiện chia:

Group A — Explicit pixel budget

Sweep:

K=\{1,2,4,8,16,32,64,128\}

với các attack có thể trực tiếp nhận K.

Đây là cách đúng để tạo:

ASR(K)

và:

RobustAccuracy(K)

cho constrained attack.

⸻

Group B — Minimal-support attacks

Ý tưởng hiện tại là:

1. chạy attack một lần để tìm perturbation;
2. đo L_0(x_{adv}-x);
3. tại mỗi K tính:

ASR@K =
\frac{
\#\{x:\text{clean correct, attack success, }L_0\le K\}
}{
\#\{\text{clean correct}\}
}

Đây chính là cumulative ASR@K mà trước đó chúng ta đã thảo luận.

Về concept, cách này tốt.

⸻

Group C — Dense/non-pixel-budget

FGSM/BIM/PGD/SFA được tách khỏi K sweep.

Cũng hợp lý vì không nên cố ép dense L_\infty attack vào cùng trục K.

⸻

4. P0: metric distortion hiện đang bị tính sai về mặt nghiên cứu

Đây là lỗi quan trọng nhất mình thấy.

Benchmark hiện cộng:

total_l0 += l0_per.sum()
total_l2 += ...
total_psnr += ...
total_ssim += ...

trên tất cả ảnh.

Sau đó:

Avg L0 = total_l0 / total_count
PSNR = total_psnr / total_count
...

Giả sử attack 10 ảnh:

* thành công 2 ảnh với L_0=10;
* thất bại 8 ảnh và trả lại ảnh gốc với L_0=0.

Repo báo:

AvgL_0 = \frac{10+10}{10}=2

Trong khi perturbation thành công thực sự là:

AvgL_{0,\ success}=10.

PSNR/SSIM còn bị bias mạnh hơn vì failed samples trả ảnh gốc sẽ có:

PSNR = 100
SSIM = 1
LPIPS ≈ 0

compute_per_sample_psnr() thực tế chủ động gán ảnh giống hệt nhau thành 100 dB.

Đây là lý do một số kết quả nhìn “quá đẹp”

Ví dụ Sparse-RS ở K rất nhỏ có thể:

* ASR > 0;
* nhưng Avg L_0 cực thấp;
* PSNR rất cao.

Không nhất thiết vì attack tạo perturbation cực đẹp, mà do các ảnh thất bại với L_0=0 kéo average xuống.

Nên báo cả hai

Mình khuyên đổi thành:

All-sample metrics
    Avg L0 all
    Avg L2 all
    ...
Successful-only metrics
    Success Avg L0
    Success Median L0
    Success Avg L∞
    Success PSNR
    Success SSIM
    Success LPIPS

Trong paper, distortion chính nên là success-conditioned.

⸻

5. Group B còn bị lỗi metric này nặng hơn

Trong Group B có:

sub_mask = (l0_arr <= K)

rồi tính:

PSNR = psnr_arr[sub_mask].mean()
SSIM = ...
Avg L0 = ...

Nhưng failed attack trả ảnh gốc:

L_0=0

vẫn thỏa:

0\le K.

Nên failed samples được tính vào quality metric.

Phải đổi thành

success_at_k = fooled_arr & (l0_arr <= K)

và quality:

quality_mask = success_at_k

không phải chỉ:

l0 <= K

Đây là P0 trước khi chạy full benchmark.

⸻

6. Conditional ASR hiện lại được tính đúng

Điểm này repo làm tốt:

adv_succ += (c_mask & (~r_mask)).sum()
asr = adv_succ / clean_correct

Tức:

ASR =
P(f(x_{adv})\ne y\mid f(x)=y)

chứ không tính những ảnh model vốn đã classify sai là attack success.

Đây là metric mình khuyên giữ.

Nên rename cột rõ hơn thành:

Conditional ASR (%)

thay vì generic:

ASR (%)

⸻

7. Kết quả đang commit chưa có giá trị thống kê cho paper

Trong main của benchmark:

num_samples = ... else 10

tức chạy script không truyền tham số thì chỉ benchmark 10 ảnh.

Report hiện tại cũng có các ASR như:

0%
10%
20%
30%
...
100%

và Clean Acc = 100% liên tục, hoàn toàn phù hợp với việc đang chạy tập 10 ảnh.

Ví dụ hiện có:

OnePixel K=1 → 10%
K=2 → 30%
...
Sparse-PGD K=4 → 60%

Nhưng với n=10:

Một ảnh = 10 percentage points.

Không thể dùng chúng để kết luận:

“Attack A mạnh hơn Attack B.”

Smoke test: tốt.

Scientific result: chưa.

⸻

8. Cách chọn sample cũng chưa tốt

Benchmark dùng:

Subset(test_ds, range(num_samples))

tức lấy:

ảnh 0 ... ảnh N-1

Không random và cũng không stratify.

Đối với final benchmark, nên tạo một index set cố định:

benchmark_indices_seed42.json

Ví dụ:

* CIFAR10 test = 10,000;
* evaluate final toàn 10,000 nếu compute cho phép;
* hoặc development = 1,000 stratified fixed images;
* tất cả attack dùng chính xác cùng sample IDs.

Quan trọng hơn nữa: lưu riêng:

clean_correct_indices

để mọi attack có cùng denominator cho conditional ASR.

⸻

9. "Official adapters" hiện hơi gây hiểu nhầm

Đây là vấn đề mình nghĩ reviewer/reproducer sẽ quan tâm.

Ngay cả khi:

use_official_adapters=True

Group A vẫn dùng custom implementation cho khá nhiều method.

Ví dụ:

"CornerSearch": CornerSearchAttack(...)

dù repo đã import:

CornerSearchOfficialAdapter

Tương tự JSMA, OnePixel, SAIF, BruSLe, IPFSA… vẫn là implementation nội bộ.

Nên log:

>>> Running Group A with Official Author Adapters

hiện tại hơi quá mạnh.

Mình khuyên mỗi result row có:

Implementation
Source
Source commit
Official
Modified
Hyperparameter profile

Ví dụ:

Attack	Implementation
PGD0	official-adapter
Sparse-RS	official-adapter
CornerSearch	custom-reimplementation
JSMA	custom-reimplementation
Proposed	ours

Như vậy minh bạch hơn rất nhiều.

⸻

10. Official PGD0 adapter có vấn đề fairness về runtime

PGD0 official wrapper làm:

GPU tensor
→ CPU
→ NumPy NHWC
→ official code
→ NumPy
→ GPU tensor

Trong khi nhiều proposed attack chạy native PyTorch CUDA.

Nếu dùng:

Time/Img

để nói proposed nhanh hơn PGD0 thì không hoàn toàn fair.

Runtime đang đo:

* algorithm;
* Python overhead;
* CPU/GPU transfers;
* implementation quality.

Nên paper báo:

wall-clock runtime

nhưng không dùng nó một mình để kết luận computational efficiency.

Thêm:

gradient evaluations
forward passes
backward passes
queries

sẽ tốt hơn.

⸻

11. Black-box benchmark đang thiếu metric quan trọng nhất: Query Count

Trong research plan, repo đã xác định đúng rằng black-box cần:

Query Number

Nhưng benchmark engine hiện chỉ output:

Avg Iterations
Time/Img

không có:

Queries
Queries-to-success
Median queries
ASR@query-budget

Đối với:

* Sparse-RS;
* OnePixel;
* BruSLe;
* Pixle;

đây là thiếu sót lớn.

Một Sparse-RS 90% ASR bằng 10,000 query và attack khác 85% bằng 500 query không thể xem là so sánh đơn giản 90 vs 85.

Nên có

ASR(Q)

với:

Q = 100
500
1,000
5,000
10,000

hoặc query budget phù hợp với original papers.

⸻

12. Stochastic attacks cần nhiều seed

Hiện:

set_seed(seed)

được gọi một lần và seed mặc định là 42.

Reproducibility như vậy tốt cho một run.

Nhưng với:

* OnePixel;
* Sparse-RS;
* BruSLe;
* các random-start optimizer;

paper nên chạy ít nhất:

3\text{–}5\ seeds

và báo:

mean\pm std

hoặc median/IQR cho runtime/query.

⸻

13. Model training có một lỗi augmentation khá đáng chú ý

dataset_loader đã đưa augmentation vào training dataset:

RandomCrop(...)
RandomHorizontalFlip()
ToTensor()

Sau đó train_clean_resnet18() lại cache toàn training dataset bằng cách iterate dataset đó.

Tức mỗi image bị:

RandomCrop + Flip

một lần trước khi cache.

Sau đó trong training loop lại áp:

train_aug = RandomCrop + RandomHorizontalFlip

mỗi epoch.

Nói cách khác:

Original
   ↓
Random augmentation #1
   ↓
CACHE vào VRAM
   ↓
Random augmentation #2 mỗi epoch

Đây không phải pipeline CIFAR training thông thường.

Nên đổi

Cache:

ToTensor only

rồi mỗi epoch:

RandomCrop + Flip

một lần.

Vừa đúng protocol hơn vừa dễ reproduce.

⸻

14. Model checkpoint chưa được pin immutable

find_existing_checkpoint() ưu tiên download từ:

Cuong2004/AA
models/resnet18_cifar10_best.pth

trên Hugging Face.

Vấn đề là cùng path này trong tương lai có thể chứa checkpoint khác.

Experiment hôm nay và 3 tuần sau có thể silently sử dụng model khác.

Paper artifact nên lưu:

checkpoint SHA256
HF revision / commit
training seed
best epoch
clean accuracy
dataset split hash

⸻

15. Requirements hiện chưa đủ để clone-and-run

requirements.txt hiện chỉ có các package cơ bản như torch, torchvision, numpy, scipy, pandas…

Nhưng source còn import:

datasets
huggingface_hub
scikit-learn
lpips

và notebook/tooling còn có dependency khác.

Tức:

pip install -r requirements.txt
python ...

không đảm bảo chạy.

Đây là P0 cho reproducibility.

Ngoài ra dùng:

torch>=2.0
numpy>=...

quá rộng cho paper artifact.

Nên có ít nhất:

requirements.txt
requirements-lock.txt

hoặc conda environment pinned.

⸻

16. Root repo đang thiếu README quan trọng

Top-level hiện có:

docs/
notebook/
src/
tests/
third_party/
...

nhưng không có README ở root.

Đối với research repository, README cần trả lời ngay:

Paper là gì?
Benchmark protocol là gì?
Cài thế nào?
Checkpoint ở đâu?
Chạy smoke test thế nào?
Chạy final benchmark thế nào?
Official implementation nào?
Output nằm ở đâu?

Hiện kiến thức này bị phân tán trong docs/.

⸻

17. Third-party directory tốt về reproducibility nhưng cần quản lý provenance

Một điểm tốt:

Bạn đã vendor code của nhiều official implementation vào third_party/.

Điều này tránh upstream repo thay đổi.

Nhưng hiện third_party/ khá lớn, có:

* source code;
* checkpoint;
* PDF;
* image;
* TensorFlow checkpoint;
* model weights.

Nên thêm:

THIRD_PARTY.md

với:

Method	Original repo	Commit SHA	License	Modifications

Đặc biệt quan trọng trước khi public code kèm paper.

⸻

18. Test suite: hướng đúng nhưng chưa test scientific correctness

Hiện test khá tốt ở mức contract:

* exact top-k;
* L_0\le K;
* output shapes;
* proposed attacks respect K;
* PSNR/SSIM shapes.

Official adapter tests cũng đã xuất hiện.

Nhưng đây chủ yếu là:

“code chạy và không vượt K.”

Chưa chứng minh:

“algorithm được implement đúng.”

Nên thêm regression tests kiểu:

same seed → same results
successful adversarial:
    clean_pred == y
    adv_pred != y
    L0 <= K
failed adversarial:
    success flag == False
ASR denominator test
cumulative ASR@K monotonic test
custom-vs-official sanity test
query counter test
metric successful-only test

⸻

19. Đánh giá riêng các Proposed Methods

Đây là phần mình thấy cần đặc biệt thận trọng.

CPA

Implementation hiện tại tính:

grad_mag
local_coop = avg_pool_3x3(grad_mag)
score = grad_mag + λ * local_coop

rồi top-K.

Về toán học nó gần với:

gradient saliency + local spatial smoothing

hơn là thực sự tính cooperative interaction giữa pixels.

Nếu paper nói:

“modeling cooperative pixel coalitions”

reviewer hoàn toàn có thể hỏi:

Coalition interaction nằm ở đâu?

Ablation bắt buộc

grad only
grad + 3×3 average
grad + Gaussian smoothing
CPA

Nếu CPA không khác đáng kể các smoothing baseline thì novelty yếu.

⸻

20. FCSA còn rõ hơn về mismatch tên–thuật toán

FCSA hiện:

grad_mean = abs(grad).mean(channel)
grad_max  = abs(grad).max(channel)
coalition_score = grad_mean * grad_max

Đây là một pixel saliency score kết hợp thống kê các RGB channels.

Nó không thực sự evaluate joint contribution của một coalition S theo dạng:

F(S)-\sum_iF(i)

hay Shapley/interaction score nào.

Tên:

Functional Coalition Sparse Attack

hiện mạnh hơn mathematical mechanism thực tế.

Nếu chọn FCSA làm proposed chính, mình nghĩ cần redesign đáng kể.

⸻

21. HSA có vấn đề claim lớn nhất

Docstring nói:

constructs a hypergraph, nodes = pixels, hyperedges = receptive fields, minimum coalition search…

Nhưng implementation thực tế là:

grad_mag
+
avg_pool_3x3(grad_mag)
+
avg_pool_5x5(grad_mag)

rồi top-K.

Không có:

* hypergraph adjacency/incidence matrix;
* actual hyperedges;
* hyperedge weights;
* message passing;
* minimum coalition algorithm;
* hypergraph optimization.

Về bản chất hiện tại mình sẽ gọi nó:

Multi-scale Neighborhood Gradient Saliency Attack

hợp lý hơn là Hypergraph Sparse Attack.

Nếu submit paper với claim “hypergraph”

Đây có thể thành một reviewer objection rất mạnh.

⸻

22. FMSA là proposed candidate mình thấy đáng phát triển nhất

FMSA có concept thực sự khác:

1. hook representation ở layer4;
2. lấy clean feature z(x);
3. optimize để tăng:

\|z(x_{adv})-z(x)\|_2^2;

4. dùng gradient để chọn sparse support;
5. hard-project về L_0\le K.

Concept này có câu chuyện nghiên cứu rõ hơn CPA/FCSA/HSA hiện tại.

Nhưng vẫn có 3 điểm cần sửa.

A. Architecture dependent

Nó hard-code:

if hasattr(model, "layer4")

Không phù hợp nếu sau này benchmark:

* ViT;
* WRN có naming khác;
* custom architecture.

Nếu không có layer4, code silently fallback về CE.

Không nên silent như vậy.

Nên có:

FeatureExtractorAdapter(model, layer_name)

⸻

B. Feature drift chưa đồng nghĩa misclassification

Bạn đang maximize:

\|z_{adv}-z_{clean}\|_2

nhưng feature đi xa không đảm bảo decision boundary bị vượt.

Mình sẽ dùng objective kiểu:

L =
L_{\text{margin}}
+
\lambda L_{\text{feature}}

hoặc:

L =
CE(f(x_{adv}),y)
+
\lambda D(z_{adv},z_{clean}).

Sau đó ablation:

CE only
Feature only
CE + Feature

sẽ rất thuyết phục.

⸻

C. Potential forward-hook lifecycle issue

Mỗi FMSA instance register:

model.layer4.register_forward_hook(...)

và dựa vào __del__() để remove.

Benchmark Group A lại tạo một attacker mới ở mỗi K.

Python không đảm bảo destructor chạy ngay lập tức, nên hook lifecycle cần quản lý rõ.

Nên:

with FeatureHook(...) as hook:
    ...

hoặc explicit:

try:
   ...
finally:
   attacker.remove_hook()

⸻

23. Một inconsistency nhỏ nhưng đáng sửa: SAIF naming

Code gọi SAIF là:

Sparsity-Aware Iterative Fast Attack

Trong research plan lại mô tả khác về acronym/characterization.

Với baseline paper, tên method, citation, formulation và source code phải tuyệt đối nhất quán.

Mình khuyên tạo:

attack_registry.yaml

mỗi method có:

name:
paper:
year:
venue:
url:
official_repo:
implementation:
constraint:
whitebox:
blackbox:
targeted:
budget_mode:

⸻

24. Defense hiện tại chưa thực sự tồn tại

Đây là một vấn đề tương đối nghiêm trọng so với scope trong docs/plan.md.

Plan đặt mục tiêu khá lớn:

* preprocessing;
* transformations;
* adversarial training;
* defense analysis.

Nhưng source tree hiện không có:

src/defenses/

trong khi run_defense_benchmark.py import:

from defenses.preprocessing.gaussian_blur ...
from defenses.preprocessing.median_filter ...
...

Nó sẽ fail import.

Chưa hết, script còn:

model = get_model("resnet18", pretrained=False)

mà không load trained checkpoint.

Nghĩa là ngay cả khi thêm defense module, benchmark hiện sẽ attack một random ResNet18.

Ngoài ra:

JSMAAttack(model, max_pixels=15)

trong defense runner, nhưng constructor thật của JSMA là:

JSMAAttack(model, k=15, ...)

=> còn một API mismatch nữa.

Defense hiện nên đánh dấu TODO / disabled, tránh khiến người đọc nghĩ nó đã hoàn thành.

⸻

25. Một protocol final mình khuyên dùng

Thay vì tiếp tục thêm method, mình sẽ khóa benchmark thành:

Development

CIFAR-10
ResNet18
1000 fixed stratified test samples
3 seeds cho stochastic attacks
K = 1,2,4,8,16,32,64,128

Dùng để debug/tune engineering.

Final paper

CIFAR-10 full 10,000 test
+ CIFAR-100 nếu đủ compute
ResNet18
+ WRN/ResNet50 cho cross-model confirmation
same fixed samples
same checkpoint
same attack objective
same K definition

Report mỗi K:

Conditional\ ASR@K

Robust\ Accuracy@K

L_0,\ L_2,\ L_\infty

PSNR,\ SSIM,\ LPIPS

trên successful adversarial examples, cộng:

runtime
gradient evaluations
queries for black-box
success-conditioned iterations/queries

Đồng thời tạo:

ASR-K\ curve

và một scalar summary:

AUC_{ASR-K}.

⸻

26. Thứ tự sửa repo mình đề xuất

Nếu mục tiêu là nhanh chóng đưa repo đến mức đáng tin cho paper, thứ tự nên là:

P0 — sửa trước khi chạy lại experiment: sửa success-conditioned distortion metrics; sửa Group B mask; chuyển benchmark sample từ range(N) sang fixed stratified indices; thêm query counter cho black-box; ghi provenance official/custom trên từng attack; sửa training double augmentation; hoàn chỉnh requirements; disable/sửa defense runner hiện đang broken.

P1 — trước khi tạo bảng paper: benchmark full 10k CIFAR-10; stochastic attacks ≥3 seeds; lưu checkpoint SHA/dataset indices/git commit/config; validation official adapters; thêm query/gradient evaluations; tách smoke-test results khỏi final results; root README + third-party provenance.

P2 — nâng chất lượng research: thu gọn attack set thành các baseline thực sự quan trọng; chọn một proposed method; làm ablation mạnh; thêm cross-model/cross-dataset; sau đó mới làm defense và transferability.

⸻

Nhận xét cuối cùng

Repo hiện tại có một điểm rất tốt: khung benchmark đã hình thành khá rõ và spatial L_0 đã được chuẩn hóa đúng.  Việc cần làm bây giờ không phải mở rộng thêm từ 20 attack lên 25 attack.

Mình sẽ ưu tiên chuyển nó từ:

“nhiều attack chạy được”

sang:

“ít nhất 8–12 attack nhưng mọi comparison đều defensible trước reviewer.”

Về proposed method, FMSA hiện là hướng mình thấy đáng đầu tư nhất trong bốn candidate, CPA có thể giữ làm auxiliary/ablation; còn FCSA và đặc biệt HSA hiện cần redesign nếu muốn giữ các claim “coalition/hypergraph”.

Nếu mục tiêu tiếp theo là sửa repo trước khi chạy full benchmark, mình có thể chuyển toàn bộ nhận xét trên thành một checklist kiểu P0/P1/P2 theo từng file cụ thể, chỉ rõ file → function → lỗi → cách sửa → expected output, để triển khai lần lượt.