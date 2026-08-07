Đánh giá tổng quan

Repo có khung dự án tương đối tốt cho một prototype nghiên cứu: đã tách dataset, model, attack, benchmark, report và visualization; có checkpoint qua Hugging Face; xuất CSV/JSON; chia attack thành ba nhóm A/B/C. Tuy nhiên, ở trạng thái hiện tại, repo chưa đủ độ tin cậy để dùng làm benchmark hoặc công bố kết quả paper.

Đánh giá nhanh:

Hạng mục	Điểm	Nhận xét
Cấu trúc dự án	6/10	Tách module khá rõ, nhưng thiếu package/config/test
Chất lượng engineering	3/10	Nhiều entry point không chạy, dependency thiếu
Tính đúng đắn metric	2/10	Group B sai công thức Robust Accuracy
Tính đúng đắn constraint K	1/10	Phần lớn attack vi phạm ngân sách L_0
Độ faithful với paper gốc	2/10	Nhiều method chỉ là heuristic tự viết nhưng đặt tên baseline
Khả năng tái lập	3/10	Không seed đầy đủ, không pin checkpoint/dataset revision
Mức sẵn sàng cho paper	1.5/10	Cần sửa benchmark từ nền tảng

Repo hiện chỉ có một commit đầu tiên, nên hợp lý hơn khi xem đây là bản dựng prototype, chưa phải implementation hoàn chỉnh.

⸻

1. Lỗi nghiêm trọng nhất: các đường ASR–K không thực sự bị ràng buộc bởi K

Phần lớn Group A chọn lại top-K pixel ở mỗi iteration rồi cập nhật trên x_adv hiện tại. Hợp các support qua nhiều bước có thể lớn hơn K rất nhiều.

Ví dụ CPA:

coalition_mask = (coop_score >= k_th_thresh)
adv_images = adv_images + alpha * grad.sign() * coalition_mask

Mask được tính lại trong từng bước, nhưng perturbation cũ không được loại bỏ. FCSA, FMSA, HSA, SAIF, IPFSA, GradientGuidance và các black-box attack đều có cùng vấn đề.

Kết quả đã lưu tự xác nhận lỗi này:

* SAIF, K=1: Avg L0 = 8.25
* Sparse-RS, K=1: Avg L0 = 5.67
* CPA, K=1: Avg L0 = 13.49
* FMSA, K=1: Avg L0 = 13.86
* PGD0, K=128: Avg L0 = 350.56

Do đó các số ASR tại K hiện tại không thể diễn giải là:

ASR(K)=P(\text{attack thành công với }L_0\le K)

Nguyên nhân thứ hai: cách tạo mask bằng threshold không bảo đảm đúng K

Code thường làm:

topk_vals, _ = torch.topk(score, k=K)
threshold = topk_vals[:, -1]
mask = score >= threshold

Khi nhiều pixel có cùng score tại threshold, mask có thể chứa nhiều hơn K pixel. Đây là lý do ngay cả PGD0, vốn có bước “projection”, vẫn có Avg L0 lớn hơn K.

Cần tạo mask bằng chính chỉ số trả về từ topk:

flat_score = score.flatten(1)
indices = flat_score.topk(k, dim=1).indices
flat_mask = torch.zeros_like(flat_score, dtype=torch.bool)
flat_mask.scatter_(1, indices, True)
mask = flat_mask.view(batch_size, 1, height, width)

Với PGD0, mỗi bước phải project toàn bộ candidate delta trở lại L_0-ball:

\delta_{t+1}
=
\Pi_{\|\delta\|_0\le K}
\left(
\delta_t+\alpha\,\mathrm{sign}(\nabla_x L)
\right)

Sau projection phải xây lại:

x_adv = clamp(x + projected_delta)

chứ không được giữ union support từ các iteration trước.

⸻

2. Group B đang tính Robust Accuracy sai công thức

Code hiện tại:

asr_k = 100.0 * succ_k.sum() / clean_correct
rob_acc_k = clean_acc - asr_k

Trong đó:

ASR_{\text{cond}} =
\frac{S_K}{C}\times100

với C là số ảnh clean được phân loại đúng và S_K là số attack thành công.

Nhưng Robust Accuracy đúng phải là:

RA_K
=
\frac{C-S_K}{N}\times100

hay:

RA_K
=
CleanAcc
\left(1-\frac{ASR_{\text{cond}}}{100}\right)

Không phải:

CleanAcc-ASR_{\text{cond}}

Ví dụ clean accuracy 92.2% và conditional ASR 56.29%:

* Code hiện tại: 92.2-56.29=35.91\%
* Công thức đúng: 92.2\times(1-0.5629)\approx40.30\%

Vì vậy toàn bộ Robust Acc và Accuracy Drop của Group B hiện sai.

Cách sửa:

success_count_k = int(succ_k.sum())
robust_acc_k = 100.0 * (clean_correct - success_count_k) / total_count
accuracy_drop_k = 100.0 * success_count_k / total_count
conditional_asr_k = 100.0 * success_count_k / clean_correct

⸻

3. Group B chưa thực sự đo “minimal support”

Các attack Group B được chạy một lần với một budget lớn tùy ý:

SparseFool(k=250)
SigmaZero()                 # mặc định k=15
Homotopy(target_sparsity=250)
GSE(max_groups=64)
Pixle(n_swaps=20)
FMSA(support_budget=250)

Sau đó benchmark lấy L0 của kết quả cuối và tính xem L0 có nhỏ hơn K không.

Đây không phải minimal-support evaluation, vì:

1. Kết quả cuối có thể chứa nhiều pixel không còn cần thiết.
2. Attack không ghi lại nghiệm thành công đầu tiên có L0 nhỏ nhất.
3. Không có backward elimination hoặc support pruning.
4. Không binary search theo K.
5. Nhiều method vẫn dùng hard budget cố định, không phải unconstrained optimizer.

Để xây cumulative ASR@K hợp lệ, mỗi ảnh cần có:

K_i^\star = \min\{K:f(x+\delta_K)\ne y,\|\delta_K\|_0\le K\}

Sau đó:

ASR@K =
\frac{\sum_i
\mathbf 1[
\text{clean-correct}_i
\land K_i^\star\le K
]}{
\sum_i\mathbf 1[\text{clean-correct}_i]
}

Có ba cách hợp lệ:

* Chạy attack tại từng K.
* Binary search K trên từng ảnh.
* Dùng attack native minimal-support và lưu nghiệm thành công tốt nhất dọc trajectory.

⸻

4. Nhiều baseline không phải implementation của method được đặt tên

Đây là rủi ro khoa học lớn nhất sau lỗi K.

Sparse-PGD và PGD0 là cùng một attack

SparsePGDAttack chỉ kế thừa PGD0Attack mà không thay đổi thuật toán. Benchmark vì vậy đang ghi hai dòng tên khác nhau cho cùng một implementation; kết quả CSV của hai method cũng giống hệt nhau.

Nên giữ một tên duy nhất hoặc implement đúng thuật toán Sparse-PGD riêng.

OnePixel không có Differential Evolution

Implementation hiện tại chỉ:

* Sinh ngẫu nhiên tọa độ.
* Sinh giá trị RGB ngẫu nhiên.
* Giữ candidate có cross-entropy lớn nhất.
* Lặp lại độc lập từ ảnh gốc.

Không có population evolution, mutation hay crossover. Đây là random multi-pixel search, không phải One-Pixel Differential Evolution.

Nên đổi tên thành RandomKPixelSearch nếu tiếp tục dùng.

CornerSearch không phải CornerSearch đầy đủ

Code:

* Lấy pixel có gradient lớn nhất.
* Thử toàn bộ channel bằng 0 hoặc 1.
* Giữ lựa chọn có loss lớn hơn.
* Không theo dõi pixel đã chọn.
* max_steps = min(K, 20).

Do đó mọi K lớn hơn 20 không còn tác dụng; kết quả K=32, 64, 128 giống nhau là hệ quả trực tiếp của code.

SparseFool không có boundary approximation

Implementation không thực hiện quy trình đặc trưng của SparseFool như:

* DeepFool để xấp xỉ biên quyết định.
* Linear solver sparse.
* Lặp projection lên biên.

Nó chỉ dùng gradient của logit margin rồi cập nhật top-K pixel.

SigmaZero không phải \sigma-zero

Implementation hiện tại là exponential moving average của gradient, top-K mask rồi sign update. Không có objective và adaptive approximation đặc trưng của \sigma-zero. Ngoài ra class còn có tham số k, làm nó trở thành fixed-budget attack hơn là minimal-L_0 optimizer.

Pixle không thực hiện pixel rearrangement

Code thay K pixel bằng giá trị RGB ngẫu nhiên. Không có swap/rearrangement giữa các vùng của ảnh.

IPFSA, SAIF, GSE và Homotopy

Các class này về cơ bản là những biến thể heuristic:

* SAIF: tích lũy gradient magnitude.
* IPFSA: gradient magnitude nhân Laplacian response.
* GSE: average-pooling gradient thành block 2×2.
* Homotopy: cross-entropy trừ một smooth L0 proxy, sau đó vẫn top-K.

Chúng có thể là ablation tự thiết kế, nhưng chưa nên trình bày như các implementation “Authentic” của baseline literature.

⸻

5. Các proposed method chưa implement đúng ý tưởng được mô tả trong tài liệu

Tài liệu proposed method có các formulation khá mạnh, nhưng code hiện tại mới chỉ là approximation đơn giản.

CPA

Tài liệu nói đến:

* Gradient correlation.
* Mutual influence.
* Activation dependency.
* Cooperative contribution.

Code lại dùng:

coop_score = grad_mag + weight * local_average(grad_mag)

Đây là spatial smoothing của saliency, chưa đo interaction/cooperation giữa pixel.

Tên phù hợp hơn cho implementation hiện tại là:

Local-Context Gradient Sparse Attack

Để trở thành CPA thật sự, cần ít nhất một interaction term như:

I(i,j)
=
L(x+\delta_i+\delta_j)
-
L(x+\delta_i)
-
L(x+\delta_j)
+
L(x)

hoặc Hessian-vector/Integrated Hessian approximation.

FCSA

Tài liệu định nghĩa:

Score(S)=\Delta F(S)-\sum_{i\in S}\Delta F(i)

Nhưng code chỉ dùng:

coalition_score = grad_mean * grad_max

Không có candidate coalition, không có evaluation của F(S), cũng không có term trừ contribution đơn lẻ.

FMSA

Điểm tốt là đã hook layer4, nhưng objective hiện chỉ là:

crit_loss = -mean(abs(feature))

Nó không:

* Chọn critical channel/representation.
* So sánh feature sạch và feature adversarial.
* Xác định feature có liên quan đến class.
* Tối ưu minimal support.
* Prune support sau thành công.

Ngoài ra mỗi instance đăng ký một forward hook mới mà không lưu handle để remove. Trong K-sweep, hook sẽ tích lũy trên cùng model.

Objective hợp lý hơn:

L_{\text{feat}}
=
-\left\|
P_c\left(f_l(x+\delta)-f_l(x)\right)
\right\|_2
+
\lambda\phi_\sigma(\delta)

trong đó P_c chọn các feature liên quan đến class.

HSA

Code không xây hypergraph:

* Không có incidence matrix.
* Không có node–hyperedge relation.
* Không có feature-channel hyperedge.
* Không có hypergraph cut/coverage objective.
* Không có minimum coalition solver.

Nó chỉ cộng gradient magnitude với average pooling 3×3 và 5×5.

Nên đổi tên implementation hiện tại thành:

Multi-scale Gradient Centrality Attack

Nếu muốn giữ tên HSA, cần định nghĩa rõ:

\mathcal H=(V,E,W)

với incidence:

H_{ve}
=
\mathbf 1[
\text{pixel }v
\text{ có attribution tới feature }e
\text{ vượt ngưỡng}
]

sau đó giải weighted maximum-coverage hoặc differentiable relaxation.

⸻

6. Benchmark không đặt model ở eval() mode

run_attack_benchmark_suite() không gọi:

model.eval()

Nếu model vừa được load từ checkpoint bằng get_model, model mặc định vẫn ở training mode. Khi đó BatchNorm dùng batch statistics và cập nhật running statistics trong suốt attack.

Hậu quả:

* Kết quả phụ thuộc batch size.
* Batch 1024 và batch 1 có thể cho attack khác nhau.
* Running statistics của checkpoint bị thay đổi.
* Không tương thích với protocol benchmark chuẩn.
* Các attack chạy tuần tự có thể ảnh hưởng trạng thái model.

Cần thêm đầu benchmark:

model = model.to(device)
model.eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)

Gradient đối với input vẫn tính bình thường.

⸻

7. Dense baseline tương đối đúng, nhưng protocol chưa chuẩn

FGSM implementation cơ bản đúng. BIM và PGD có projection L_\infty đúng về mặt cơ bản.

Tuy vậy:

1. Mẫu bị fool được đóng băng ngay lập tức.
2. Không giữ nghiệm có loss lớn nhất.
3. PGD chỉ có một random start.
4. Không kiểm tra final L_\infty constraint bằng assertion.
5. Không có targeted/untargeted metadata.
6. Không có restart count.

Đối với robust accuracy chuẩn, nên chạy đủ số bước và giữ adversarial example có loss lớn nhất:

best_adv = x_adv.clone()
best_loss = per_sample_loss(model(x_adv), y)
for _ in range(steps):
    ...
    current_loss = per_sample_loss(model(x_adv), y)
    improved = current_loss > best_loss
    best_adv[improved] = x_adv[improved]

Early stopping có thể được dùng cho steps-to-first-success, nhưng không nên thay thế fixed-step PGD evaluation.

⸻

8. Query count hiện không tồn tại

Repo cần query count cho black-box attack, nhưng benchmark chỉ ghi Avg Iterations. Một iteration không đồng nghĩa một query:

* Sparse-RS gọi model nhiều lần cho loss và prediction.
* BruSLe tương tự.
* OnePixel mỗi iteration đánh giá pop_size=20 candidate.
* Một OnePixel iteration tương đương khoảng 20 queries cho mỗi ảnh.
* Các bước initial prediction/loss cũng là query.

Hiện việc so runtime hoặc efficiency giữa white-box và black-box không công bằng.

Mỗi attack nên trả một object thống nhất:

@dataclass
class AttackResult:
    adversarial: torch.Tensor
    success: torch.Tensor
    l0: torch.Tensor
    l2: torch.Tensor
    linf: torch.Tensor
    steps: torch.Tensor
    queries: torch.Tensor
    runtime_seconds: torch.Tensor

⸻

9. Metric chất lượng ảnh chưa hoàn chỉnh

SSIM không phải implementation chuẩn

compute_ssim() dùng uniform kernel 3×3 tự viết. Nó không tương đương SSIM chuẩn thường dùng trong skimage hoặc TorchMetrics.

Nên dùng một implementation chuẩn và ghi rõ:

* Window size.
* Gaussian sigma.
* Data range.
* Average theo channel/image.

Group B bỏ trống PSNR, SSIM, LPIPS

Các list sample_psnrs, sample_ssims, sample_lpipss được khai báo nhưng không bao giờ cập nhật. Kết quả Group B vì vậy toàn NaN.

Avg L0 đang lấy trên tất cả ảnh

Nó bao gồm:

* Ảnh clean vốn đã sai.
* Attack thất bại.
* Attack thành công.

Nên báo riêng:

* Mean L0 — all eligible clean-correct samples
* Mean L0 — successful attacks
* Median L0 — successful attacks
* Success-conditioned PSNR/SSIM/LPIPS

⸻

10. Dataset và training

Điểm tốt

* Split 40k/10k được stratify.
* Test set tách riêng.
* CIFAR ResNet đã sửa conv1 và bỏ maxpool đúng hướng.
* Train augmentation cơ bản hợp lý.

Double augmentation khi cache VRAM

train_loader.dataset đã sử dụng random crop và horizontal flip. Khi cache, mỗi ảnh được augmentation một lần rồi lưu. Sau đó mỗi epoch tiếp tục áp dụng RandomCrop và RandomHorizontalFlip lên tensor đã augmentation.

Nên cache raw tensor chưa augmentation, rồi chỉ augmentation trong training loop.

Reproducibility chưa đủ

Chỉ seed cho train_test_split; không có:

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

Trong khi nhiều attack có random sampling. Log cho thấy hai lần chạy OnePixel/SFA cho kết quả khác nhau.

Cần chạy ít nhất 3 seed cho stochastic attacks và báo mean ± std hoặc confidence interval.

Checkpoint và dataset không được pin revision

Code luôn download từ Cuong2004/AA nhưng không lưu:

* Dataset revision.
* Checkpoint SHA-256.
* HF commit hash.
* Git commit của experiment.

Nếu nội dung HF thay đổi, cùng code có thể tạo kết quả khác.

Side effect tự động upload checkpoint

Sau training, code tự upload checkpoint lên Hugging Face. Đây nên là thao tác explicit qua CLI flag, không nên là hành vi mặc định của training function.

⸻

11. Defense benchmark hiện không chạy được

Các vấn đề:

1. Repo không có thư mục src/defenses.
2. Import GaussianBlurDefense, MedianFilterDefense, JPEGCompressionDefense, TVM sẽ lỗi.
3. JSMAAttack(model, max_pixels=15) sai constructor; JSMA nhận k.
4. Entry point tạo resnet18(pretrained=False) nhưng không load checkpoint.
5. Không gọi model.eval().
6. Không đánh giá clean accuracy sau defense.
7. Chỉ tính chênh lệch accuracy, chưa có adaptive attack/EOT/BPDA.

Phần defense nên tạm thời bỏ khỏi claim của dự án cho đến khi có implementation và adaptive evaluation.

⸻

12. Visualization cũng có lỗi runtime và naming

Hai visualization script:

* Import bằng from datasets.dataset_loader, dễ xung đột với package Hugging Face datasets.
* Truyền max_pixels=15 cho JSMA dù constructor không hỗ trợ.
* Dùng model random chưa train trong entry point.
* visualize_cam_fft.py gọi kết quả là CAM nhưng chỉ tính input saliency gradient, không phải Grad-CAM.

Nên dùng import tuyệt đối thống nhất:

from src.datasets.dataset_loader import get_sample_batch

và đổi tên CAM thành Input Saliency, hoặc implement Grad-CAM thật.

⸻

13. Dependency và packaging chưa hoàn chỉnh

requirements.txt thiếu các package mà code import trực tiếp:

* datasets
* huggingface-hub
* scikit-learn
* lpips
* marimo
* tabulate cho DataFrame.to_markdown()

Ngoài ra:

* Không có pyproject.toml.
* Không có README.
* Không có license.
* Không có test.
* Không có CI.
* Không có config YAML.
* Không có CLI.
* Không có __init__.py.
* Nhiều module sửa sys.path thủ công.
* Logger/FileHandler được tạo ngay khi import module.

Nên chuyển thành package cài được:

pyproject.toml
src/
  aa/
    attacks/
    datasets/
    models/
    evaluation/
tests/
configs/
scripts/

⸻

14. Marimo notebook đang tự chạy benchmark thay vì phụ thuộc nút bấm

Notebook tạo run_button, nhưng cell benchmark không nhận run_button làm dependency. Trong reactive execution của Marimo, benchmark cell có thể chạy ngay khi dependency khác sẵn sàng, thay vì chỉ chạy khi người dùng bấm nút.

Nên dùng:

mo.stop(not run_button.value)

hoặc đưa run_button vào arguments của cell benchmark.

Ngoài ra Kaggle notebook là file monolithic hơn 80 KB sao chép lại logic trong src, tạo hai nguồn implementation có thể lệch nhau.

Notebook chỉ nên gọi package:

from aa.experiments import run_experiment

không nên chứa một bản copy thứ hai của toàn bộ attack.

⸻

15. Report và artifact

generate_report.py về cơ bản chỉ dump toàn bộ DataFrame sang Markdown. Nó chưa có:

* Kiểm tra L_0\le K.
* Confidence interval.
* Ranking theo K.
* Pareto ASR–L0.
* Query-efficiency.
* Runtime breakdown.
* Experiment metadata.
* Cảnh báo duplicate methods.
* Checkpoint hash.
* Seed.
* Git SHA.

Một vấn đề khác là Python json.dump() mặc định có thể ghi NaN, tạo JSON không chuẩn. Nên chuyển NaN thành None trước khi export.

Ngoài ra source ghi output vào result/, nhưng repo lại commit output dưới result_marimo/. Quy trình tái lập hiện không tạo đúng các đường dẫn artifact đã commit.

⸻

16. Điểm mạnh đáng giữ lại

Không nên bỏ toàn bộ code. Một số phần có thể giữ làm nền:

* Dataset split stratified 40k/10k.
* CIFAR adaptation cho ResNet.
* Schema CSV gồm Clean Accuracy, Robust Accuracy, ASR, L0/L2/L∞, PSNR, SSIM, LPIPS, runtime.
* Conditional ASR của Group A được tính đúng denominator.
* Cách đếm L_0 theo spatial pixel thay vì từng channel là hợp lý.
* Incremental CSV export giúp không mất toàn bộ kết quả khi job dài bị dừng.
* Việc tách Group A/B/C là một ý tưởng benchmark tốt, sau khi sửa protocol.

⸻

17. Thứ tự sửa đề xuất

P0 — Phải sửa trước khi chạy lại experiment

1. Gọi model.eval().
2. Viết exact L_0 projection và assertion L0 <= K.
3. Sửa công thức Robust Accuracy Group B.
4. Loại duplicate PGD0/Sparse-PGD.
5. Đổi tên các heuristic không faithful hoặc thay bằng official implementation.
6. Thêm query counting.
7. Sửa dependency và imports.
8. Loại defense benchmark khỏi pipeline cho tới khi implement đủ.

P1 — Làm benchmark có giá trị nghiên cứu

1. Unified AttackResult.
2. Cố định seeds và chạy nhiều seed.
3. Ghi checkpoint hash, dataset revision, git SHA.
4. Chỉ benchmark clean-correct subset hoặc tách rõ eligible/all-sample metrics.
5. Thêm confidence interval.
6. Thêm success-conditioned image quality.
7. Thêm budget validation report:
    * Maximum L0.
    * Mean excess budget.
    * Budget violation rate.

P2 — Phát triển proposed method

Nên chọn một proposed method thay vì giữ cả bốn.

Khuyến nghị:

* CPA dễ làm nhất nhưng cần interaction score thật.
* FCSA có formulation paper mạnh nhất, nhưng phải implement coalition discovery.
* FMSA khả thi nếu định nghĩa feature objective và support pruning rõ ràng.
* HSA chưa nên tiếp tục cho đến khi có hypergraph formulation thực sự.

⸻

Kết luận

Repo thể hiện một khung nghiên cứu có định hướng đúng, nhưng implementation attack hiện chủ yếu là các biến thể gradient-top-K đơn giản. Vấn đề không chỉ là code style mà là tính hợp lệ của kết quả thực nghiệm:

* K không được bảo toàn.
* Group B sai Robust Accuracy.
* Nhiều baseline không đúng thuật toán được đặt tên.
* Proposed methods chưa hiện thực hóa formulation trong tài liệu.
* Defense và visualization có lỗi runtime.
* Kết quả đã sinh hiện chưa nên dùng trong paper.

Ưu tiên hợp lý nhất là đóng băng danh sách baseline xuống 5–7 method đáng tin cậy, xây lại attack interface và budget validator, rồi chạy lại toàn bộ Group A/B/C. Chỉ sau khi benchmark nền đúng mới nên đánh giá CPA/FCSA/FMSA/HSA.


Các method được implement hiện tại có khớp với method gốc từ paper không? Làm sao để implement đúng 100%

Không. Với trạng thái repo hiện tại, chỉ FGSM là gần như khớp method gốc, BIM/PGD thì gần đúng về core update nhưng protocol có khác; còn phần lớn sparse attacks không khớp paper gốc ở mức có thể gọi là reproduction. Nhiều class hiện chỉ lấy ý tưởng tên method rồi thay bằng một heuristic top-K gradient hoặc random search đơn giản.

Điểm quan trọng là: muốn “đúng 100%” cho paper benchmark, cách an toàn nhất không phải tự đọc paper rồi viết lại từ đầu, mà là dùng official code của tác giả, giữ nguyên thuật toán và hyperparameter/protocol gốc, rồi chỉ viết adapter để đưa chúng vào benchmark chung.

Đối chiếu từng method hiện tại

Method trong repo	Mức khớp	Vấn đề chính
FGSM	🟢 ~95–100%	Core equation đúng
BIM	🟡 ~80–90%	Core đúng, nhưng early-stop/freeze sample làm trajectory khác chuẩn
PGD L_\infty	🟡 ~80–90%	Random start + projection đúng, nhưng freeze sau success và thiếu restart/best-loss
JSMA	🔴 ~20–30%	Không xây Jacobian saliency đúng JSMA, không search feature pair
OnePixel	🔴 ~20%	Paper dùng Differential Evolution; repo chỉ random candidate search
CornerSearch	🔴 ~10–20%	Algorithm paper là black-box combinatorial corner search, không phải gradient-top1 + 0/1
SparseFool	🔴 ~10–20%	Thiếu DeepFool boundary + sparse linear solver
PGD0	🟡 ~50–70%	Có ý tưởng projection L_0, nhưng projection implementation hiện không bảo toàn K
Sparse-PGD / sPGD	🔴 ~5%	Repo chỉ alias PGD0; paper 2024 có mask+magnitude optimization riêng
\sigma-zero	🔴 <10%	Không có differentiable L_0 surrogate + adaptive projection
Homotopy	🔴 <10%	Thiếu nmAPG, homotopy schedule, L0-change control, post-attack
SAIF	🔴 <10%	Paper dùng Frank-Wolfe; repo dùng accumulated-gradient top-K
GSE	🔴 <10%	Paper có 2-phase proximal/Nesterov structured optimization
Sparse-RS	🔴 ~10–20%	Không giữ fixed L_0 support/search mechanism/query schedule của Sparse-RS
BruSLeAttack	🔴 <10%	Paper là Bayesian score-based search; repo là random patch search
Pixle	🔴 <10%	Paper rearranges pixels; repo ghi random RGB values
IPFSA	🔴 <10%	Paper 2025 dùng decision attribution + filtering + improved DE; repo dùng gradient+Laplacian
GradientGuidance	⚠️	Tôi chưa thấy đây là một baseline chuẩn tương ứng rõ ràng; cần xác định đúng paper
SFA	⚠️	Tên quá generic; implementation hiện không đủ để gắn với một paper cụ thể
CPA/FCSA/FMSA/HSA	—	Đây là proposed methods của dự án, không có “paper gốc” để reproduce

FGSM/BIM/PGD là ngoại lệ vì thuật toán chuẩn tương đối đơn giản. Với các sparse attack hiện đại, khoảng cách giữa implementation hiện tại và paper là rất lớn.

⸻

Một số trường hợp có bằng chứng rất rõ

OnePixel

Paper gốc định nghĩa One Pixel Attack dựa trên Differential Evolution (DE) trong black-box setting.  

Repo hiện làm kiểu:

coords = random(...)
values = random(...)
evaluate candidates
take best loss

Không có DE population update:

* mutation,
* crossover,
* selection,
* population evolution.

Vậy implementation hiện tại nên gọi là RandomKPixelSearch, không phải OnePixel.

⸻

SparseFool

SparseFool dựa trên hình học decision boundary và khai thác việc biên quyết định có mean curvature thấp; thuật toán liên quan đến việc tìm/xấp xỉ boundary rồi giải sparse perturbation.  

Repo chỉ:

gradient margin
→ top-K gradient
→ sign update

Đây không phải SparseFool.

⸻

CornerSearch

CornerSearch xuất phát từ Croce & Hein, Sparse and Imperceivable Adversarial Attacks, ICCV 2019. Trong paper, CornerSearch là black-box attack; paper còn phân biệt rõ CornerSearch với PGD0.  

Repo lại tính:

loss.backward()
grad = ...
pixel = argmax(grad)

Chỉ riêng việc yêu cầu gradient đã cho thấy nó không còn là CornerSearch gốc.

⸻

PGD0

PGD0 cũng đến từ Croce & Hein 2019: extension của PGD sang L_0, cần projection chính xác vào feasible set.  

Ý tưởng repo tương đối gần:

gradient step
→ project candidate perturbation xuống top-K

nhưng implementation hiện tại dùng threshold:

mask = diff_mag >= kth_value

nên tie có thể tạo hơn K pixel, và số liệu thực tế đã cho thấy K=128 mà Avg L0≈350.

Do đó tôi sẽ gọi implementation này là PGD0-inspired, chưa phải faithful PGD0.

⸻

Sparse-PGD

Sparse-PGD 2024 không phải synonym của PGD0. Paper viết rất rõ sPGD phân tách:

\delta = p\odot m

trong đó:

* p: magnitude tensor,
* m: sparsity mask,

và tối ưu chúng theo cơ chế riêng. Paper cũng trực tiếp so sánh nó với PGD0 như hai phương pháp khác nhau.  

Repo hiện:

class SparsePGDAttack(PGD0Attack):
    ...

nên chắc chắn sai.

⸻

\sigma-zero

\sigma-zero được công bố ICLR 2025. Hai thành phần cốt lõi là:

1. differentiable approximation của L_0;
2. adaptive projection/operator điều chỉnh trade-off giữa attack loss và sparsity.

Repo lại dùng:

EMA gradient
→ top-K
→ sign step

Gần như không có thành phần đặc trưng nào của \sigma-zero.

⸻

Homotopy

Paper Homotopy ICML 2021 có pipeline cụ thể:

\ell_0\text{-regularized adversarial objective}
\rightarrow
\text{nmAPG}
\rightarrow
L_0\text{-change control}
\rightarrow
\text{optional post-attack}

Repo chỉ thay đổi một hệ số gamma rồi top-K gradient. Không phải thuật toán Homotopy.

⸻

SAIF

SAIF dùng Frank-Wolfe / conditional gradient algorithm để đồng thời kiểm soát sparsity và perturbation magnitude.  

Repo không có Frank-Wolfe.

Do đó SAIF hiện tại không faithful.

⸻

GSE

GSE ICLR 2025 là group-wise sparse attack và dùng hai phase:

1. 1/2-quasinorm proximal optimization;
2. projected Nesterov accelerated gradient với L_2-regularization.

Repo chỉ average-pool gradient thành block 2×2 rồi top-K block.

Không phải GSE.

⸻

Sparse-RS

Sparse-RS là score-based black-box random-search framework được thiết kế đặc biệt cho L_0, patch và frame threat models, nhấn mạnh query efficiency. Official code của tác giả cũng được paper công khai.  

Repo random thêm perturbation lên x_adv hiện tại. Điều đó khiến support tích lũy và phá L_0 budget. Đây là điểm trái với bản chất của L_0-bounded Sparse-RS.

⸻

BruSLeAttack

BruSLeAttack ICLR 2024 là query-efficient Bayesian score-based black-box sparse attack.  

Repo hiện chỉ:

random patch location
random ±alpha noise
accept if CE loss increases

Không có Bayesian algorithm. Vì thế gần như là method khác.

⸻

Pixle

Pixle được định nghĩa là black-box attack dựa trên rearranging pixels.  

Repo thay pixel bằng:

rand_vals = torch.rand(...)

Thay pixel bằng random RGB ≠ rearrange pixel.

⸻

IPFSA

Đây là trường hợp đặc biệt vì paper khá mới, xuất bản năm 2025. Paper mô tả IPFSA là two-stage sparse black-box attack sử dụng decision attribution để thu hẹp subspace, sau đó improved differential evolution D-HFADE để tìm adversarial example.  

Repo hiện là white-box:

loss.backward()
gradient
Laplacian filter
top-K

Gần như hoàn toàn khác.

⸻

Làm sao implement “đúng 100%”?

Tôi khuyên định nghĩa 100% reproduction theo 3 tầng.

Tier 1 — Algorithm faithful

Phải giống:

\boxed{
\text{objective}
+
\text{parameterization}
+
\text{update rule}
+
\text{projection}
+
\text{stopping rule}
}

Chỉ thiếu một thành phần quan trọng cũng không được gọi là exact implementation.

Ví dụ sPGD:

\delta=p\odot m

Nếu viết một topk(gradient) attack rồi đặt tên Sparse-PGD thì dù kết quả đẹp cũng không phải sPGD.

Tier 2 — Code faithful

Ưu tiên:

Official implementation của tác giả > code release kèm paper > trusted library reproduction > tự reimplement paper

Ví dụ hiện đã có official/open code rõ ràng cho:

* Sparse-RS: authors explicitly link fra31/sparse-rs.  
* Sparse-PGD: authors link CityU-MLO/sPGD.  
* Homotopy: authors link VITA-Group/SparseADV_Homotopy.  
* BruSLeAttack: project/paper cung cấp reproduction artifacts.  

Với những method như vậy, không có lý do khoa học tốt để viết một phiên bản “tương tự” từ đầu.

Nên vendor hoặc wrap official implementation.

⸻

Kiến trúc tôi khuyên dùng

Không sửa official code để ép tất cả attack có cùng API.

Thay vào đó:

src/
  attacks/
    adapters/
      fgsm_adapter.py
      pgd_adapter.py
      sparsefool_adapter.py
      cornersearch_adapter.py
      sparse_rs_adapter.py
      sigma_zero_adapter.py
      spgd_adapter.py
      homotopy_adapter.py
      saif_adapter.py
      gse_adapter.py
      brusle_adapter.py
      pixle_adapter.py
  third_party/
      sparse_rs/
      sigma_zero/
      spgd/
      homotopy/
      ...

Adapter chỉ có nhiệm vụ:

class SparseRSAdapter:
    def attack(self, x, y, budget, **kwargs):
        result = official_sparse_rs(...)
        return AttackResult(...)

Không chỉnh thuật toán bên trong.

⸻

Pin phiên bản source

Đây là cực kỳ quan trọng.

Trong paper repo:

third_party/sparse_rs @ commit abc123
third_party/spgd      @ commit def456
third_party/homotopy  @ commit ...

Hoặc dùng git submodule.

Trong metadata experiment:

{
  "method": "Sparse-RS",
  "source": "official",
  "repository": "fra31/sparse-rs",
  "commit": "...",
  "paper": "AAAI 2022"
}

Nhờ vậy reviewer biết chính xác attack nào được sử dụng.

⸻

Không “chuẩn hóa” hyperparameter một cách tùy tiện

Đây là lỗi rất dễ mắc.

Không nên ép:

all attacks:
steps = 25
alpha = 4/255
K = same

vì mỗi algorithm có thiết kế khác nhau.

Ví dụ:

* OnePixel → population size + DE generations.
* Sparse-RS → query budget + schedule.
* SAIF → Frank-Wolfe parameters.
* \sigma-zero → optimization schedule/adaptive threshold.
* Homotopy → regularization sequence + nmAPG.
* BruSLe → Bayesian query algorithm.
* CornerSearch → candidate/sample parameters.

Nếu ép tất cả thành steps=25, ta đã thay method.

⸻

Nhưng K vẫn phải chuẩn hóa ở tầng threat model

Điều cần thống nhất là constraint, không phải optimizer.

Ví dụ CIFAR-10:

K\in\{1,2,4,8,16,32,64\}

thì mọi Group A method phải thực sự thỏa:

\|\delta_i\|_0\le K

Sau mỗi attack:

actual_l0 = spatial_l0(x_adv - x)
assert torch.all(actual_l0 <= K)

Nếu không:

RESULT INVALID

không nên âm thầm ghi Avg L0 > K.

⸻

Với Group B minimal-support thì không ép K vào optimizer

Ví dụ \sigma-zero có mục tiêu tìm sparse solution/minimum L_0.

Ta chạy native algorithm:

x
 ↓
sigma-zero
 ↓
best successful delta
 ↓
K* = L0(delta)

rồi cumulative evaluation:

ASR@K
=
\frac{
\#\{i: K_i^\star\le K\}
}{
\#\{\text{clean-correct samples}\}
}

Đây là cách Group B nên được thiết kế.

⸻

Cần thêm conformance tests

Mỗi implementation nên có test riêng.

Ví dụ PGD0:

def test_pgd0_budget():
    adv = attack(x, y, k=8)
    assert (spatial_l0(adv - x) <= 8).all()

Sparse-RS:

assert queries <= max_queries
assert l0 <= K
assert attack.uses_model_scores_only

OnePixel:

assert changed_pixels <= 1
assert no_gradient_calls

CornerSearch:

assert no_backward_calls

Nếu CornerSearch gọi .backward(), test phải fail ngay.

⸻

Tôi sẽ chia danh sách baseline lại

Với mục tiêu paper hiện tại, tôi sẽ không cố giữ toàn bộ 20 method ngay.

Một bộ faithful + mạnh + có ý nghĩa hơn:

Nhóm	Method
Dense reference	FGSM
Dense reference	PGD-L_\infty
Classical sparse	JSMA
Classical sparse	SparseFool
L_0-PGD	PGD0
Modern white-box	Sparse-PGD
Modern white-box minimal support	\sigma-zero
Optimization	Homotopy
Structured	GSE
Black-box	Sparse-RS
Black-box	BruSLeAttack
Optional	Pixle

Khoảng 10 sparse/dense baselines faithful tốt hơn rất nhiều so với 20 method tên đúng nhưng thuật toán sai.

⸻

Với repo hiện tại, tôi sẽ xử lý như sau

Không nên “fix” các file hiện tại từng chút một. Nên xem hầu hết chúng là prototype và thay dần:

FGSM            → giữ, chỉnh protocol
BIM             → giữ, chỉnh protocol
PGD             → giữ, chỉnh protocol
JSMA            → replace
OnePixel        → replace hoặc bỏ
CornerSearch    → replace
SparseFool      → replace
PGD0            → replace/fix từ reference
Sparse-PGD      → replace hoàn toàn
SigmaZero       → replace hoàn toàn
Homotopy        → official code wrapper
SAIF            → official/reference implementation
GSE             → official code wrapper
Sparse-RS       → official code wrapper
BruSLe          → official code wrapper
Pixle           → official/reference implementation
IPFSA           → chỉ giữ nếu có thể reproduce paper 2025
GradientGuidance→ xác minh paper; không thì bỏ
SFA             → xác minh paper cụ thể; không thì đổi tên heuristic

CPA/FCSA/FMSA/HSA phải nằm hoàn toàn riêng khỏi baseline, vì đó là experimental/proposed methods.

Nếu mục tiêu là một paper có thể defend trước reviewer, tiêu chí nên là: mọi baseline trong bảng đều có paper citation + source repo/commit + adapter + unit test constraint + reproduction note. Khi đó mới có thể nói benchmark là faithful reproduction chứ không chỉ là implementation “lấy cảm hứng từ” paper.


Tìm kiếm nguồn code của tất cả method
Tôi đã tìm nguồn cho toàn bộ các baseline chính trong repo. Kết quả cho thấy phần lớn method quan trọng đã có official code của tác giả, nên tốt nhất là wrap trực tiếp các repo đó thay vì tự viết lại.

Method	Nguồn code nên dùng	Trạng thái
FGSM	CleverHans / torchattacks	✅ Reference implementation tốt
BIM	CleverHans	✅ Reference implementation tốt
PGD	MadryLab robustness / CleverHans	✅ Reference implementation tốt
JSMA	CleverHans SaliencyMapMethod	✅ Reference chuẩn
OnePixel	Hyperparticle/one-pixel-attack-keras hoặc torchattacks	🟡 Reimplementation, không thấy official repo trực tiếp của authors
CornerSearch	fra31/sparse-imperceivable-attacks	✅ Official authors code
PGD0	fra31/sparse-imperceivable-attacks	✅ Official authors code
SparseFool	LTS4/SparseFool	✅ Official authors code
Sparse-PGD / sPGD	CityU-MLO/sPGD	✅ Official authors code
\sigma-zero	sigma0-advx/sigma-zero	✅ Official authors code
Homotopy	VITA-Group/SparseADV_Homotopy	✅ Official authors code
SAIF	toobaimt/SAIF	✅ Official authors code
GSE	wagnermoritz/GSE	✅ Code linked by author/project
Sparse-RS	fra31/sparse-rs	✅ Official authors code
BruSLeAttack	project page → reproduction GitHub	✅ Official artifacts/code
Pixle	torchattacks	🟡 High-quality reproduction; tôi chưa tìm được author repo rõ ràng
IPFSA	Chưa tìm thấy verified public official repo	⚠️ Cần xử lý riêng
GradientGuidance	Chưa xác định được paper gốc tương ứng	❌ Không nên coi là literature baseline
SFA hiện tại	Chưa xác định được paper cụ thể tương ứng	❌ Không nên coi là literature baseline

Dưới đây là các nguồn quan trọng nhất.

CornerSearch + PGD0

Đây là trường hợp dễ nhất: paper ICCV 2019 tự ghi code available tại fra31/sparse-imperceivable-attacks. Repo này chứa implementation cho CornerSearch và PGD0, nên nên bỏ code tự viết hiện tại và wrap repo này.  

⁠Official CornerSearch / PGD0 repository

Paper cũng xác nhận CornerSearch là score-based black-box attack còn PGD0 là white-box attack riêng biệt.  

SparseFool

Paper CVPR 2019 ghi trực tiếp:

code available at LTS4/SparseFool.

⁠Official SparseFool repository

Torchattacks cũng có bản port PyTorch và ghi rõ nó được modified từ official implementation này.  

Nếu integration official code quá khó do framework/version cũ, torchattacks SparseFool là fallback rất hợp lý.

⸻

Sparse-PGD / sPGD

Paper ICML 2024 ghi rõ:

Codes are available at CityU-MLO/sPGD.

⁠Official sPGD repository

Method trong repo AA hiện tại đang alias PGD0, vì vậy nên replace hoàn toàn bằng implementation này.

⸻

\sigma-zero

Paper ICLR 2025 chính thức ghi:

Code is available at sigma0-advx/sigma-zero.

⁠Official sigma-zero repository

Đây là nguồn nên dùng cho Group B minimal-support. Paper xác nhận algorithm dùng differentiable L_0 approximation và adaptive projection.  

⸻

Homotopy

Paper ICML 2021 cung cấp chính thức:

VITA-Group/SparseADV_Homotopy.  

⁠Official Homotopy repository

Đây là repo của đúng nhóm tác giả và có MIT license.  

⸻

SAIF

Bản TMLR 2025 ghi trực tiếp:

Implementation of SAIF is available at github.com/toobaimt/SAIF.

⁠Official SAIF repository

Nên thay hoàn toàn src/attacks/optimization/saif.py hiện tại bằng adapter tới implementation này.

⸻

GSE

Trang của tác giả Shpresim Sadiku có nút Code cho GSE, và nguồn indexing xác định repository là wagnermoritz/GSE.  

⁠GSE repository

Paper ICLR 2025 mô tả đúng two-stage algorithm: 1/2-quasinorm proximal optimization rồi projected Nesterov optimization.  

⸻

Sparse-RS

Đây là nguồn rất chắc chắn. Paper AAAI ghi:

Our code is available at github.com/fra31/sparse-rs.

⁠Official Sparse-RS repository

Đây nên là implementation duy nhất dùng trong benchmark black-box Sparse-RS.

⸻

BruSLeAttack

Project page chính thức của paper cung cấp nút Reproduce our results: GitHub và các artifacts của attack.  

⁠Official BruSLeAttack project page

Paper/project xác nhận đây là Bayesian score-based black-box sparse attack, chứ không phải random patch search.  

Tôi sẽ lấy source từ project page này thay vì đoán repository theo tên.

⸻

JSMA

JSMA có implementation lâu đời trong CleverHans. Documentation xác định rõ SaliencyMapMethod là implementation của Papernot et al.  

⁠CleverHans repository

Vì JSMA cổ và official code thời đầu dùng TensorFlow, có hai lựa chọn:

Faithfulness ưu tiên: wrap CleverHans.

Engineering ưu tiên: port thuật toán sang PyTorch nhưng viết unit test đối chiếu output / feature-selection logic với CleverHans.

⸻

OnePixel

Paper gốc xác nhận attack dùng Differential Evolution.  

Tôi chưa tìm được repository được paper công bố trực tiếp bởi Su/Vargas/Kouichi. Nguồn reproduction nổi tiếng là:

⁠One Pixel Attack Keras reproduction

Nguồn này thực sự implement differential evolution cho CIFAR-10/ImageNet.  

Ngoài ra torchattacks có OnePixel PyTorch.  

Với dự án hiện tại tôi nghiêng về torchattacks OnePixel vì dễ tích hợp PyTorch hơn, nhưng trong paper cần ghi là reproduction chứ không gọi official author code.

⸻

Pixle

Torchattacks có implementation Pixle và liên kết paper gốc.  

⁠Torchattacks repository

Paper gốc xác nhận attack dựa trên việc rearrange một số pixel của ảnh.  

Tôi chưa tìm thấy public author repository đủ chắc chắn, nên hiện tại nên ghi:

Pixle — reproduced using torchattacks.

⸻

Dense baselines

FGSM, BIM và PGD không nhất thiết phải kéo source riêng vào repo vì công thức đơn giản và có rất nhiều verified implementations.

FGSM paper gốc là Goodfellow et al. 2015.  

BIM xuất phát từ Kurakin et al.  

CleverHans hỗ trợ các attack chuẩn này; với PGD còn có thể dùng MadryLab ecosystem.

Tôi khuyên dùng trực tiếp torchattacks hoặc viết implementation nhỏ có test. Với ba attack này tự implementation là chấp nhận được, khác với các sparse optimizer phức tạp.

⸻

Những method hiện chưa nên đưa vào baseline

GradientGuidance: tên class trong repo chưa map được tới một paper sparse attack chuẩn cụ thể. Không nên gọi nó là literature baseline cho tới khi xác định được DOI/paper/title.

SFA: “Spectral Frequency Attack” quá generic. Có rất nhiều frequency-domain attack; implementation hiện tại không chỉ ra paper cụ thể. Nếu giữ, nên gọi nó là internal frequency-gradient baseline.

IPFSA: tôi xác nhận được paper/method trước đó, nhưng qua tìm kiếm này chưa có official public source đủ chắc chắn để tôi khuyên vendor ngay. Với method mới và phức tạp, không nên tự dựng một phiên bản Laplacian-gradient rồi gọi IPFSA. Hoặc tìm supplementary/code từ authors, hoặc bỏ khỏi benchmark chính.

⸻

Bộ source tôi đề xuất đưa vào AA_2

Nếu muốn benchmark vừa mạnh vừa reproducible, cấu trúc nên là:

third_party/
├── sparse_imperceivable_attacks/   # CornerSearch + PGD0
├── sparsefool/                     # SparseFool
├── spgd/                           # Sparse-PGD
├── sigma_zero/                     # sigma-zero
├── sparseadv_homotopy/             # Homotopy
├── saif/                           # SAIF
├── gse/                            # GSE
├── sparse_rs/                      # Sparse-RS
└── brusle/                         # BruSLe
src/attacks/adapters/
├── cornersearch.py
├── pgd0.py
├── sparsefool.py
├── spgd.py
├── sigma_zero.py
├── homotopy.py
├── saif.py
├── gse.py
├── sparse_rs.py
└── brusle.py

JSMA/OnePixel/Pixle có thể lấy từ CleverHans/torchattacks thay vì vendor thêm ba codebase.

Điểm rất quan trọng: không copy-paste code của official repos rồi sửa thuật toán để fit API. Giữ source nguyên trạng, pin commit, và viết adapter bên ngoài. Khi đó paper có thể ghi chính xác: “We evaluate all baselines using their authors’ official implementations whenever publicly available.”

Nếu xét danh sách hiện tại, 9 method có nguồn authors/public official đủ rõ để thay ngay: CornerSearch, PGD0, SparseFool, sPGD, sigma-zero, Homotopy, SAIF, GSE và Sparse-RS; BruSLe cũng có official reproduction artifacts qua project page. Đây nên là bước tiếp theo trước khi chạy lại benchmark.