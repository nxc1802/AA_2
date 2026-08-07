Tôi đã rà main hiện tại của nxc1802/AA_2 ở mức source code, benchmark runner, tests, notebook, kết quả đã commit và tài liệu thiết kế. Đây là static code review; tôi chưa chạy end-to-end toàn bộ benchmark trên GPU, nên các nhận xét về bug đến từ luồng code và sự không nhất quán nội bộ.

Kết luận tổng quan

Repo đã tiến xa hơn một prototype notebook thông thường: cấu trúc src/ khá rõ, có core utilities cho spatial L_0, benchmark engine, official adapters, tests, report generator và tách riêng proposed methods. Root hiện có docs/, notebook/, src/, tests/, third_party/, kết quả Marimo và requirements.

Nhưng nếu đánh giá theo tiêu chuẩn research code dùng để lấy số cho paper, tôi xếp repo hiện tại khoảng:

Thành phần	Đánh giá
Kiến trúc source code	7.5/10
Core L_0 / projection	8.5/10
Dense baselines	8/10
Sparse baseline fidelity	4/10
Benchmark methodology	5.5/10
Proposed implementation ↔ formulation	3/10
Reproducibility	5/10
Tests	4.5/10
Defense	1/10
Paper-ready tổng thể	~4.5–5/10

Điểm quan trọng nhất là: framework tốt hơn chất lượng implementation của một số attack bên trong. Nếu sửa đúng các vấn đề P0 bên dưới thì repo có nền tảng khá tốt để trở thành benchmark research nghiêm túc.

⸻

1. Kiến trúc repo: tốt

src/ được chia thành attacks, benchmark, core, datasets, models, reports, visualization, khá đúng cách tổ chức một research framework.

Taxonomy attack cũng hợp lý:

Nhóm	Hiện có
Dense	FGSM, BIM, PGD
Classical sparse	JSMA, OnePixel, CornerSearch, SparseFool
Optimization	PGD0, Sparse-PGD, SigmaZero, Homotopy, GSE, SAIF
Black-box	Sparse-RS, BruSLe, Pixle
Attribution/Frequency	IPFSA, GradientGuidance, SFA
Proposed	CPA, FCSA, FMSA, HSA
Official adapter	Sparse-RS, CornerSearch, PGD0, SparseFool, SigmaZero, sPGD, Homotopy, GSE

Đặc biệt, việc thêm third_party/ + adapter là đúng hướng hơn rất nhiều so với reimplement toàn bộ SOTA bằng code tự viết. Các official adapters hiện đã có cho tám method.

Một điểm tốt nữa là notebook Marimo đã trở thành wrapper gọi code trong src/ thay vì chứa toàn bộ logic bên trong.

Tuy nhiên Kaggle script vẫn là file đơn khoảng 80 KB với một implementation độc lập, tức hiện tại vẫn tồn tại hai source of truth: modular src/ và legacy single-file Kaggle.

⸻

2. Core L_0: đây là phần mạnh nhất

compute_spatial_l0() định nghĩa pixel bị sửa nếu bất kỳ RGB channel nào tại vị trí (h,w) thay đổi. Đây chính là cách nên dùng khi paper nói “K pixels”, thay vì đếm từng scalar RGB thành ba phần tử.

exact_spatial_topk_mask() cũng rất đáng khen: dùng index Top-K thay vì threshold, tránh trường hợp các saliency score bằng nhau làm mask chứa nhiều hơn K pixel.

project_l0() chọn top-K spatial positions dựa trên magnitude qua các channel và zero phần còn lại. Với các method có hard K-budget, đây là primitive rất hữu ích.

Các test cũng kiểm tra:

* exact K mask;
* project_l0 không vượt K;
* PGD0/sPGD/Sparse-RS/BruSLe/proposed budget;
* PSNR/SSIM shape.

Đây là nền tảng tốt để enforce một threat model chung.

⸻

3. Mười vấn đề quan trọng nhất cần sửa

1. Implementation của proposed methods chưa tương ứng với formulation trong tài liệu. Đây là vấn đề lớn nhất cho paper. proposed_method.md mô tả coalition interaction, set function, feature-to-minimal-support và hypergraph optimization rất mạnh. Nhưng code CPA/FCSA/HSA phần lớn vẫn là gradient saliency → spatial smoothing/scoring → Top-K → PGD update.
2. Không phải mọi baseline mang tên một paper đều là implementation trung thực của paper đó. Custom JSMA chẳng hạn thực tế dùng gradient của chênh lệch logits, lấy pixel có magnitude lớn và tăng RGB; nó không phải classical Jacobian Saliency Map formulation đầy đủ.  Custom Sparse-RS cũng chỉ random lại toàn bộ K support từ clean image mỗi vòng rồi giữ candidate nếu loss tăng, khác khá xa một Sparse-RS chuẩn.  Với benchmark paper, official adapter nên là mặc định.
3. Cùng một repo nhưng Marimo và benchmark CLI có thể chạy baseline khác nhau. run_attack_benchmark_suite() mặc định use_official_adapters=False; __main__ của runner truyền True, nhưng Marimo gọi hàm mà không truyền tham số này. Như vậy chạy Marimo sẽ dùng custom implementations, còn chạy file benchmark trực tiếp lại dùng official implementations.    Hai đường chạy có thể tạo hai bảng số hoàn toàn khác nhau.
4. Một số Group A không thực sự giữ semantics K như bảng benchmark nói. BruSLe factory truyền block_size=floor(sqrt(K)), trong khi class lại đặt effective k=block\_size^2. Vì vậy requested K=8 thực tế thành K=4, K=32 thành 25, K=128 thành 121.   JSMA lại có max_iter=25, vì vậy sweep K=32,64,128 không thể thực sự sử dụng budget tương ứng.
5. Có một số bug thuật toán cụ thể. OnePixelAttack tính outs cho population ban đầu nhưng sau khi differential-evolution selection không cập nhật lại outs; success check ở iteration sau tiếp tục dùng prediction cũ. Điều này làm early success và last_steps sai, thậm chí có thể ảnh hưởng candidate được chọn.  Custom Sparse-PGD còn nghiêm trọng hơn: m_logits khởi tạo bằng zero → Top-K chọn arbitrary support → gradient của p bên ngoài support bằng zero → mask score bên ngoài cũng không tăng. Nói cách khác support ban đầu rất dễ bị “đóng băng”.
6. IPFSA, GradientGuidance và FMSA có nguy cơ final L_0>K. Mỗi iteration chúng chọn Top-K mới rồi cập nhật trực tiếp x_adv; nếu support đổi qua các vòng thì perturbation cũ không bị xoá. Không có project_l0(x_adv-x,K) cuối mỗi bước.    Test hiện tại chưa kiểm tra IPFSA/GradientGuidance và dummy model có thể vô tình giữ gradient ranking ổn định nên không bắt được FMSA overflow.
7. Group A và Group B không dùng cùng định nghĩa Robust Accuracy. Group A/C thực sự tính adv_preds == y trên toàn bộ test subset. Group B lại tính clean_correct - success_count_k. Hai công thức không hoàn toàn tương đương nếu clean-misclassified samples bị attack làm trở lại đúng.   ASR@K của Group B là ý tưởng đúng, nhưng Robust Accuracy phải thống nhất toàn benchmark.
8. Metric quality của Group B tại từng K hiện chưa có ý nghĩa. ASR@K được tính từ fooled & L0<=K, tốt. Nhưng Avg L0, L2, L∞, PSNR, SSIM, LPIPS lại lấy mean của toàn bộ output ban đầu rồi copy y nguyên vào mọi K. Vì vậy PSNR@K=1 và PSNR@K=128 của một method Group B luôn giống nhau.  Nên tính metric trên successful ∧ L0≤K, đồng thời báo cáo riêng all samples / successful samples.
9. Runtime hiện không phải attack runtime. Timer bắt đầu trước cả clean prediction và kết thúc sau adversarial prediction, L0/L2/L∞, PSNR, SSIM, LPIPS. Ngoài ra dùng time.time() trên CUDA nhưng không torch.cuda.synchronize().  Do CUDA asynchronous, Time/Img có thể thiếu chính xác và còn bao gồm chi phí metric. Black-box methods cũng chưa có query-count chuẩn.
10. Kết quả đã commit không tương ứng chắc chắn với source main hiện tại. result_marimo/full_attack_metrics.csv dùng schema cũ, chưa có Max L0; log cũ thậm chí ghi SAIF K=1 nhưng Avg L0=8.25, K=2 Avg L0=14.16, trong khi SAIF hiện tại đã có hard projection.   Nghĩa là sau lần modular upgrade, toàn bộ result nên được rerun, không nên dùng result_marimo/ hiện tại làm số cuối cho paper.

⸻

4. Dense baselines: tương đối ổn

FGSM dùng \epsilon=8/255, gradient sign và clamp [0,1].

BIM dùng \epsilon=8/255, \alpha=2/255, 10 iterations và projection về L∞ ball.

PGD thêm random initialization trong [-ε,+ε], 20 iterations và lưu best-loss adversarial example.

Đây là những baseline tôi ít lo nhất trong repo.

Tuy nhiên phải ghi rõ trong paper rằng dense baseline chạy dưới L∞ threat model, còn sparse attacks chạy L0. Không nên nói PGD “thua” một sparse method chỉ bằng ASR mà không nói chúng thuộc threat model khác nhau.

⸻

5. Vấn đề fairness của sparse benchmark

Đây là điểm quan trọng hơn cả K.

Hai attack cùng K=8 chưa chắc đang chịu constraint giống nhau.

OnePixelAttack cho RGB mới nằm bất kỳ trong [0,1].  CornerSearch đặt pixel thẳng về 0 hoặc 1.  Trong khi custom Sparse-RS chỉ dùng perturbation ±4/255.  CPA/FCSA/PGD0 tăng ±4/255 mỗi iteration và có thể tích lũy magnitude qua nhiều bước.

Do đó một bảng chỉ cố định K nhưng bỏ qua L_\infty hoặc value-domain constraint có thể không công bằng.

Bạn cần chọn rõ một trong hai protocol:

Protocol	Ý nghĩa
Pure L0	Cho phép K pixel nhận bất kỳ giá trị hợp lệ [0,1]
L0 + L∞	Giữ cùng K và cùng ε_\infty cho tất cả method có thể áp dụng

Nếu method gốc có threat model đặc thù, giữ native configuration nhưng không gộp kết luận “mạnh hơn dưới cùng constraint” nếu constraint thực tế khác.

⸻

6. Group B và cumulative ASR@K: ý tưởng đúng nhưng implementation cần chỉnh

Tôi đánh giá việc tách:

Group A = attack nhận K trực tiếp

và

Group B = attack tìm sparse solution rồi hậu kiểm L0<=K

là hợp lý hơn việc cố nhét mọi method vào direct K sweep.

Phần code:

ASR@K=
\frac{\#\{clean\ correct \land fooled \land L_0\le K\}}
{\#clean\ correct}

đang làm đúng tinh thần cumulative ASR@K.

Nhưng SigmaZeroOfficialAdapter trong Group B hiện được khởi tạo mà không truyền K, vì vậy adapter lấy default k=15 rồi truyền nó thành epsilon_budget. Như vậy nó không thật sự chạy “unconstrained minimal support” trên toàn range 1→128.

Ngoài ra fallback SigmaZero tính true_l0 trên flattened scalar tensor, tức có khả năng đếm từng channel thay vì spatial pixel, trong khi benchmark core dùng spatial L0. Đây là chỗ phải audit kỹ trước khi dùng fallback.

Homotopy adapter còn except Exception: rồi im lặng trả lại clean image.  Với research benchmark đây là hành vi nguy hiểm: lỗi integration có thể bị báo cáo thành “attack failed”. Nên fail-fast hoặc ít nhất ghi status=ERROR và loại sample đó khỏi result thay vì coi là failure hợp lệ.

⸻

7. Proposed methods: khoảng cách giữa ý tưởng và code

Đây là phần tôi sẽ sửa trước khi nghĩ tới ablation.

Method	Tài liệu tuyên bố	Code hiện tại	Nhận xét
CPA	Cooperative/interacting pixels	grad_mag + local 3×3 average → Top-K	Có ý tưởng spatial cooperation, nhưng chưa thật sự đo interaction
FCSA	Coalition set-function, joint contribution	grad_mean × grad_max từng pixel → Top-K	Không phải coalition discovery
FMSA	Feature → minimal pixel support	Maximize layer4 feature distance + Top-K gradient	Có feature objective thật, nhưng chưa có minimal-support search
HSA	Hypergraph nodes/hyperedges + minimum coalition	Gradient + 3×3/5×5 average pooling → Top-K	Chưa có hypergraph thực sự

CPA source:
FCSA source:
FMSA source:
HSA source:
Design claims:

Ví dụ FCSA documentation định nghĩa:

Score(S)=\Delta F(S)-\sum_{i\in S}\Delta F(i)

tức score của một tập pixel.

Nhưng implementation hiện tại:

score_i =
mean_c(|g_{i,c}|)
\times
max_c(|g_{i,c}|)

vẫn là score độc lập cho từng pixel i.

Reviewer chỉ cần nhìn equation và code/pseudocode là có thể chỉ ra khoảng cách này.

HSA còn rõ hơn: hiện không có incidence matrix, hyperedge membership, feature-to-pixel relationship hay hypergraph optimization nào. Hai avg_pool2d() với kernel 3 và 5 chưa đủ để gọi là hypergraph construction.

Proposed nào gần khả thi nhất?

Từ code hiện tại, FMSA là hướng gần một method distinct nhất, vì nó thật sự thay objective classification bằng objective làm lệch representation từ layer4.

Nhưng tôi sẽ chưa gọi nó “Feature-to-Minimal Support” cho tới khi bổ sung search thực sự, ví dụ:

K_0 > K_1 > K_2 > \dots

hoặc binary/adaptive support reduction để tìm:

K^*(x)=\min K\quad
\text{s.t.}\quad f(x+\delta_K)\neq y.

Nếu chỉ Top-K gradient tại fixed support_budget, tên an toàn hơn sẽ gần với Feature-Guided Sparse Attack.

CPA cũng có thể phát triển nhanh, nhưng cần một interaction term có nghĩa toán học hơn local average gradient.

FCSA/HSA hiện cần thay đổi lớn nếu muốn giữ đúng tên và novelty claim.

⸻

8. Training pipeline: khác requirement khá nhiều

requirement.md yêu cầu 40k/10k stratified split, CIFAR-10/CIFAR-100, ResNet18 và WRN-28-10, early stopping theo validation loss, patience 20, batch size 256, checkpoint tốt nhất.

Dataset split thực tế làm khá chuẩn: stratified 40k/10k, seed 42, test 10k riêng.

Nhưng model factory hiện chỉ hỗ trợ ResNet18 và ResNet50, không có WRN-28-10. Default classes cũng là 10 và checkpoint manager được viết riêng cho resnet18_cifar10_best.pth.

Training thực tế còn:

Requirement	Code
batch 256	function default 1024
Early Stopping patience 20	Không có
monitor Validation Loss	Không tính val loss
best validation loss	save best validation accuracy
validation thường xuyên	chỉ mỗi 20 epochs
WRN-28-10	chưa có
CIFAR-100 complete pipeline	chưa có

Ngoài ra có một vấn đề subtle: train_loader.dataset vốn đã dùng RandomCrop + RandomHorizontalFlip. Code load toàn dataset qua loader để cache, tức mỗi ảnh đã bị random augment một lần trước khi cache. Sau đó mỗi epoch lại áp dụng một train_aug RandomCrop + Flip lần nữa.

Nếu muốn cache VRAM, nên cache clean tensor chưa augmentation, rồi chỉ augment trong training loop.

⸻

9. Reproducibility của attack tốt hơn training

set_seed() có Python, NumPy, Torch CPU/GPU và chuyển CuDNN sang deterministic.

Attack benchmark gọi set_seed(seed) ngay đầu, đây là tốt.

Training lại không gọi set_seed() trong training function, trong khi module model ban đầu bật cudnn.benchmark=True.

Vì vậy một checkpoint train mới không chắc reproducible hoàn toàn chỉ từ source hiện nay.

⸻

10. AttackResult là thiết kế tốt nhưng chưa được sử dụng

Repo đã có:

AttackResult(adversarial, success, l0, l2, linf, steps, queries, runtime_seconds, best_loss).

Đây chính xác là interface mà toàn bộ attack nên trả về.

Nhưng hiện tất cả các attack vẫn chỉ:

return x_adv

sau đó benchmark đi tìm attacker.last_steps hoặc đoán từ steps/max_iter.

Hệ quả là:

Metric	Hiện trạng
Actual iterations	đôi khi có
Query count	chưa chuẩn
Forward count	thiếu
Backward count	thiếu
Per-sample runtime	thiếu
Failure/error status	thiếu
Requested K / actual K	rời rạc

Tôi rất khuyến nghị biến AttackResult thành contract bắt buộc. Đây sẽ là một refactor nhỏ nhưng nâng chất lượng benchmark rất mạnh.

⸻

11. Metric hiện tại: tốt ở cơ bản, thiếu ở paper-level

PSNR và SSIM được tính per-sample trong pixel domain [0,1], LPIPS cũng convert [0,1] → [-1,1].

Đây là đúng hướng.

Nhưng so với chính requirement của repo thì còn thiếu:

Requirement	Current
L0	Có
L0 ratio	Có
L1	Thiếu
L2	Có
L∞	Có
perturbation magnitude trên modified pixels	Thiếu
PSNR	Có
SSIM	Có
MS-SSIM	Thiếu
LPIPS	Optional
mean	Có
std	Thiếu
median/percentiles/distribution	Thiếu
metrics successful examples only	Thiếu
query count	Thiếu
forward/backward	Thiếu
memory	Thiếu

Requirement đầy đủ nằm ở đây.

Avg L0 Ratio hiện còn hard-code:

L_0/1024

thay vì:

L_0/(H\times W)

nên đúng cho CIFAR 32×32 nhưng không generalize.

⸻

12. Evaluation subset hiện mới phù hợp development

Attack benchmark mặc định chỉ lấy:

range(1000)

trong test set 10,000 ảnh.

Cho debug/development thì tốt.

Cho main paper result, chưa đủ.

Tôi sẽ tách rõ:

Phase	Samples
Smoke/debug	100–200
Development/ablation nhanh	fixed 1,000
Main table	full 10,000
Expensive black-box	fixed stratified subset nếu chi phí quá lớn, ghi rõ

Và thay vì “first 1000”, nên lưu một file fixed sample IDs được random-stratified bằng seed 42 để tất cả methods chạy chính xác cùng ảnh.

⸻

13. Current results chưa nên dùng làm main result

File kết quả cũ báo Clean Accuracy 92.2%, FGSM ASR 42.84%, BIM ~97%, PGD ~97.7%, cùng một loạt sparse results.

Nhưng tôi sẽ không phân tích ranking method dựa trên bảng này, vì source đã thay đổi sau lần result được sinh.

Bằng chứng rõ nhất là log cũ:

SAIF K=1 → Avg L0=8.25

trong khi current SAIFAttack đã gọi hard project_l0(..., self.k) mỗi iteration.

Do đó bảng số cũ không còn đại diện cho current implementation.

⸻

14. Tests: có nền nhưng coverage chưa đủ

Hiện có đúng hai test files.

test_attacks.py kiểm budget và metric cơ bản. test_official_adapters.py kiểm Sparse-RS, PGD0 budget và một số adapter ở mức shape.

Nhưng hiện thiếu các invariants quan trọng như:

max L0 <= requested K cho mọi Group A method và mọi K, test runner end-to-end, deterministic test, query accounting, test OnePixel evolution, test IPFSA/GG support, actual ResNet smoke test, Group B cumulative-ASR semantic test và official-vs-wrapper contract.

Đặc biệt shape-only test cho official adapter không đủ. Một adapter có thể trả tensor đúng shape nhưng sai normalization, sai budget hoặc silent-fail.

Repo root cũng chưa thấy .github/workflows, nên chưa có CI tự động chạy tests.

⸻

15. Defense hiện chưa chạy được

Đây là phần yếu nhất.

run_defense_benchmark.py import:

defenses.preprocessing.gaussian_blur, median_filter, jpeg_compression, tvm

nhưng src/ hiện không có thư mục defenses.

Nó còn gọi:

JSMAAttack(model, max_pixels=15, ...)

trong khi current JSMAAttack.__init__() dùng k, không có max_pixels.

Tệ hơn nữa, __main__ của defense benchmark chỉ:

get_model("resnet18", pretrained=False)

mà không load checkpoint trained.

Nếu sửa import mà quên điểm này thì defense sẽ được benchmark trên random network.

Ngoài ra requirement nói có robust checkpoints/RobustBench, nhưng code hiện chưa triển khai phần này.

Vì vậy: attack framework đang ở mức prototype có thể phát triển; defense framework hiện mới là skeleton.

⸻

16. Requirements/dependencies chưa reproducible

requirements.txt hiện chỉ có Torch, torchvision, numpy, scipy, matplotlib, seaborn, tqdm, Pillow, scikit-image, pandas.

Nhưng code còn trực tiếp cần:

Package	Dùng ở
datasets	dataset loader
scikit-learn	stratified split
huggingface_hub	checkpoint
marimo	Marimo notebook
lpips	perceptual metric

Dataset source xác nhận datasets + sklearn.  Model source cần Hugging Face Hub.

Một máy mới chạy pip install -r requirements.txt chưa đủ để reproduce repo.

Root cũng chưa có README hay root LICENSE trong listing hiện tại.  Với một repo có nhiều code vendored trong third_party/, nên đặc biệt bổ sung provenance, upstream URL, upstream commit và license từng method.

⸻

17. Visualization/report

Report generator đã có pivot ASR–K, RobustAccuracy–K, iterations và image metrics.

Đây là nền tảng tốt.

Nhưng visualization hiện chỉ so Original / PGD / JSMA / PGD0 trên vài ảnh.

Trong khi report paper cần ít nhất:

Panel	Nên có
A	Clean
B	Adversarial
C	|x_{adv}-x|
D	amplified perturbation
E	binary spatial L0 mask
F	clean/adv prediction + confidence

Đặc biệt proposed method cần visual mask cạnh các strongest baselines ở cùng K, nếu muốn chứng minh support có cấu trúc tốt hơn.

⸻

18. Roadmap tôi khuyên cho repo này

Ưu tiên	Việc cần làm	Lý do
P0	Bắt mọi Group A method pass Max L0 <= K	Correctness trước hết
P0	Fix BruSLe K mapping	Hiện K label ≠ effective K
P0	Fix OnePixel stale predictions	Bug thuật toán
P0	Fix custom sPGD support freeze	Bug thuật toán
P0	Project IPFSA/GG/FMSA về hard K	Budget integrity
P0	JSMA max_iter >= K hoặc bỏ K>25	Sweep hiện không hợp lệ
P0	Official adapters làm default ở mọi entrypoint	Baseline fidelity
P0	Thống nhất Group A/B Robust Acc	Metric consistency
P0	Correct Group B per-K quality metrics	PSNR@K hiện sai semantics
P0	Query counter + CUDA-synchronized attack timer	Fair efficiency comparison
P0	Rerun toàn bộ results	Old result đã stale
P1	Chỉ chọn một proposed method chính	Tập trung novelty
P1	Làm algorithm thực sự khớp formulation	Quan trọng cho reviewer
P1	Full 10k final evaluation	Main paper
P1	Multiple seeds cho stochastic attacks	Statistical reliability
P1	mean/std + successful-only quality	Paper-quality metrics
P1	Hoàn thiện defense trước khi lấy số	Defense hiện broken
P2	AttackResult contract	Clean architecture
P2	YAML/dataclass experiment config	Reproducibility
P2	Fix training pipeline/early stop	Match methodology
P2	README + requirements + CI + licenses	Release-quality
P2	Xóa duplication Kaggle	Single source of truth

⸻

19. Cấu trúc đích tôi đề xuất

Điểm tôi muốn giữ nguyên là taxonomy src/attacks/ và core projection.

Nhưng benchmark nên chuyển thành kiểu:

AttackSpec
   ↓
Attack.attack(...)
   ↓
AttackResult
   ├── adversarial
   ├── success
   ├── l0
   ├── queries
   ├── forward_count
   ├── backward_count
   ├── iterations
   └── runtime
       ↓
Evaluator
       ↓
Per-sample raw results
       ↓
Aggregator
       ├── Group A: direct K
       ├── Group B: cumulative ASR@K
       └── Group C: native threat model

Quan trọng là lưu per-sample result trước, rồi mới aggregate.

Hiện tại benchmark aggregate ngay trong loop. Nếu sau đó muốn tính median, confidence interval, success-only PSNR, cumulative ASR@K mới, hoặc kiểm tra outlier thì rất khó.

Một file dạng:

sample_id
attack
seed
requested_k
actual_l0
success
clean_correct
clean_pred
adv_pred
l1
l2
linf
psnr
ssim
lpips
queries
iterations
runtime

sẽ giải quyết phần lớn nhu cầu report sau này.

⸻

Đánh giá cuối cùng

Điều đáng giữ: cấu trúc modular, core spatial-L0, exact Top-K/projector, 3-group benchmark concept, official-adapter architecture, fixed train/val split và hướng lưu report.

Điều chưa nên tin: ranking từ result_marimo, fidelity của nhiều custom sparse baselines, budget của tất cả Group A, runtime hiện tại và toàn bộ defense results.

Rủi ro lớn nhất cho paper không phải model accuracy mà là experimental validity. Nếu một reviewer phát hiện K không thực sự giống nhau giữa methods, custom implementation khác paper gốc, hoặc formulation FCSA/HSA không được thuật toán thực thi, những vấn đề đó nặng hơn vài phần trăm ASR.

Về hướng phát triển, tôi sẽ đóng băng proposed method tạm thời, trước tiên biến baseline benchmark thành một “golden benchmark” hoàn toàn đáng tin: official implementation → exact budget → per-sample metrics → query/runtime chuẩn → full test. Sau đó mới phát triển một proposed method duy nhất trên nền đó. Khi baseline framework đã đáng tin, mọi cải thiện của proposed method mới có giá trị khoa học rõ ràng.