Được. Mình sẽ mô tả proposed method theo hướng **Coalition-Aware Sparse Adversarial Attack** từ formulation → khởi tạo → tìm coalition → tối ưu RGB → support exchange → success → giảm L0 → output, để sau này có thể chuyển gần như trực tiếp thành `proposed_v2.py`.

Tên hiện tại chỉ nên xem là working name:

[
\boxed{\textbf{CASA — Coalition-Aware Sparse Attack}}
]

Ý tưởng trung tâm là:

> Không hỏi “K pixel nào mạnh nhất khi xét riêng lẻ?”, mà hỏi “K pixel nào tạo adversarial effect mạnh nhất khi hoạt động cùng nhau?”

---

# 1. Bài toán chúng ta thực sự muốn giải

Cho classifier:

[
f(x)\in\mathbb R^C
]

với ảnh:

[
x\in[0,1]^{3\times H\times W}
]

và ground-truth label (y).

Ta muốn tìm adversarial image:

[
x_{\mathrm{adv}}=x+\delta
]

sao cho:

[
f(x_{\mathrm{adv}})\neq y
]

và chỉ thay đổi tối đa (K) spatial pixels:

[
|\delta|_{0,\text{spatial}}\le K.
]

Ta dùng attack margin:

[
J(x,y)
======

\max_{c\neq y}z_c(x)-z_y(x).
]

Nếu:

[
J(x,y)>0
]

thì sample đã bị misclassify.

Bài toán budgeted attack là:

[
\boxed{
\max_{\delta}
J(x+\delta,y)
}
]

subject to:

[
|\delta|_{0,\text{spatial}}\le K,
\qquad
x+\delta\in[0,1].
]

Điểm quan trọng là (\delta) thực ra chứa **hai quyết định khác nhau**:

[
\boxed{
\underbrace{S}*{\text{pixel nào}}
+
\underbrace{\delta_S}*{\text{pixel đó đổi thành gì}}
}
]

với:

[
S\subseteq{1,\ldots,HW},
\qquad |S|\le K.
]

Đây chính là nơi proposed mới khác current SFA.

---

# 2. Core formulation: đánh giá cả một support

Ta định nghĩa một set function:

[
\boxed{
F(S)
====

\max_{\delta:\operatorname{supp}(\delta)\subseteq S}
J(x+\delta,y)
}
]

subject to:

[
x+\delta\in[0,1].
]

Ý nghĩa:

> (F(S)) là adversarial margin tốt nhất có thể đạt được nếu ta chỉ được phép thay đổi các pixel trong tập (S).

Ví dụ:

[
F({A})=0.5,
]

[
F({B})=0.4,
]

nhưng:

[
F({A,B})=1.8.
]

Khi đó A và B phối hợp rất tốt.

Ngược lại:

[
F({C})=0.9,
\quad
F({D})=0.8,
]

nhưng:

[
F({C,D})=1.0.
]

C và D individually mạnh, nhưng redundant.

Đây là failure mode của Top-K individual importance.

---

# 3. Khái niệm quan trọng nhất: conditional gain

Thay vì score một pixel độc lập:

[
Score(j)=|g_j|,
]

ta quan tâm:

[
\boxed{
\Delta(j\mid S)
===============

F(S\cup{j})-F(S)
}
]

Đây là **marginal contribution của pixel (j) đối với coalition hiện tại (S)**.

Cùng một pixel (j) có thể:

[
\Delta(j\mid S_1)=0.1
]

nhưng:

[
\Delta(j\mid S_2)=1.4.
]

Tức pixel không có “importance tuyệt đối”.

Nó có:

> **importance conditional on the pixels already selected.**

Đây là concept nền của toàn proposed method.

---

# 4. Pairwise interaction chỉ là trường hợp đơn giản

Có thể định nghĩa:

[
I(i,j)
======

F({i,j})
-F({i})
-F({j})
+F(\varnothing).
]

Vì clean sample thường:

[
F(\varnothing)=J(x,y),
]

nên interaction dương:

[
I(i,j)>0
]

nghĩa là hai pixel có synergy.

Interaction âm:

[
I(i,j)<0
]

nghĩa là redundancy.

Nhưng final algorithm **không nên chỉ dựa vào pairwise interaction**.

Vì support có thể là:

[
S={A,B,C,D,E}.
]

Pixel (E) có thể không synergy mạnh riêng với A hay B, nhưng lại rất tốt khi kết hợp với toàn coalition:

[
\Delta(E\mid{A,B,C,D})\gg0.
]

Do đó mục tiêu cuối cùng là conditional set gain, không chỉ pair matrix.

---

# 5. Vì sao không thể tính (F(S)) brute-force?

CIFAR-10 có:

[
32\times32=1024
]

spatial positions.

Chỉ riêng pair:

[
{1024\choose2}=523{,}776.
]

Nếu K=16 thì số possible supports là:

[
{1024\choose16},
]

không thể enumerate.

Do đó challenge của proposed method là:

[
\boxed{
\text{Approximate coalition utility cheaply}
}
]

rồi chỉ exact-evaluate một số candidate tốt.

Đây chính là chỗ technical contribution nằm.

---

# 6. Toàn bộ attack gồm 8 stage

1. **Tạo candidate pool** từ tất cả pixels bằng một cheap first-order estimate.
2. **Khởi tạo support** nhỏ bằng các pixel promising.
3. **Tối ưu RGB values** trên support hiện tại.
4. **Ước lượng coalition gain** của candidate pixels ngoài support.
5. **Đo redundancy** của các pixel đang nằm trong support.
6. **Support exchange:** bỏ pixel yếu và thêm pixel bổ trợ tốt.
7. **Re-optimize + accept/reject** bằng objective thật.
8. Khi success, chạy **Drop-and-Repair** để giảm support xuống gần minimal (L_0).

Giờ mình đi chi tiết từng stage.

---

# 7. Stage 1 — Candidate generation

Ta chưa thể tính interaction của toàn 1024 pixels.

Đầu tiên dùng gradient:

[
g=
\nabla_x J(x,y).
]

Nhưng thay vì đơn giản:

[
Score(i)=|g_i|_1,
]

ta dùng **box-aware potential gain**.

Với pixel (i), gradient RGB:

[
g_i=(g_{i,R},g_{i,G},g_{i,B}).
]

Trong pure (L_0), pixel được phép di chuyển toàn range:

[
[0,1]^3.
]

Ta hỏi:

> Nếu linear approximation đúng, pixel này có thể làm margin tăng tối đa bao nhiêu?

Giải:

[
A_i
===

\max_{v_i}
g_i^\top v_i
]

subject to:

[
x_i+v_i\in[0,1]^3.
]

Closed form:

[
v^*_{ic}
========

\begin{cases}
1-x_{ic},&g_{ic}>0\
-x_{ic},&g_{ic}<0.
\end{cases}
]

Và:

[
\boxed{
A_i
===

\sum_c
g_{ic}v^*_{ic}
}
]

hay tương đương:

[
A_i=
\sum_c
|g_{ic}|
\begin{cases}
1-x_{ic},&g_{ic}>0\
x_{ic},&g_{ic}<0.
\end{cases}
]

Đây tốt hơn raw gradient vì nó xét cả **khả năng pixel thực sự di chuyển trong box**.

Ta lấy Top-M:

[
C=\operatorname{TopM}(A),
]

ví dụ:

[
M=32\text{ hoặc }64.
]

Candidate pool (C) chỉ dùng để giảm search space.

---

# 8. Stage 2 — Khởi tạo coalition

Phiên bản đơn giản có thể lấy:

[
S_0=\operatorname{TopK}(A).
]

Nhưng điều này vẫn là individual ranking.

V2 tốt hơn có thể khởi tạo greedily.

Đầu tiên:

[
i_1=\arg\max_i A_i.
]

Sau đó không lấy second-best independent pixel.

Ta chọn:

[
i_2=
\arg\max_{j\in C\setminus S}
\widehat{\Delta}(j\mid S).
]

Rồi:

[
S\leftarrow S\cup{i_2}.
]

Lặp tới khi:

[
|S|=K.
]

Nhưng (\widehat{\Delta}) là gì?

Đó là coalition gain approximation mà ta sẽ xây ở stage 4.

Trong implementation đầu tiên, mình vẫn khuyên dùng Top-K box-aware để initialization đơn giản, rồi để support exchange sửa coalition.

Như vậy dễ ablate hơn.

---

# 9. Stage 3 — Tối ưu RGB trên support cố định

Giả sử đã có:

[
S={s_1,\ldots,s_K}.
]

Bây giờ bài toán trở thành continuous:

[
\max_{\delta_S}
J(x+\delta_S,y)
]

subject to:

[
x+\delta_S\in[0,1].
]

Không cần L0 projection nữa vì chỉ pixels trong (S) có gradient/update.

Đây là một điểm rất khác current SFA.

Current SFA update toàn perturbation rồi project lại Top-K.

Ở proposed mới:

[
\boxed{\text{Support fixed} \Rightarrow \text{L0 constraint tự động được đảm bảo}}
]

Có thể dùng projected gradient ascent:

[
x^{t+1}_{S}
===========

\Pi_{[0,1]}
\left(
x^t_S+\alpha\nabla_{x_S}J
\right).
]

Hoặc Adam.

Mình nghi Adam tốt hơn sign gradient ở đây vì RGB values là continuous variables thật sự.

Sau (T_{\text{inner}}) iterations:

[
\delta_S^*
\approx
\arg\max_{\delta_S}J.
]

Ta có:

[
F(S)\approx J(x+\delta_S^*,y).
]

---

# 10. Warm-start pixel values

Khi một pixel mới (j) được add vào support, không nên initialize:

[
\delta_j=0.
]

Ta đã có analytical box-aware direction.

Do đó initialize:

[
x'_{jc}
=======

\begin{cases}
1,&g_{jc}>0\
0,&g_{jc}<0.
\end{cases}
]

hoặc một softer version:

[
x'_{jc}
=======

\operatorname{clip}
(x_{jc}+\eta,\operatorname{sign}(g_{jc}),0,1).
]

Sau đó inner optimizer refine.

Điều này đặc biệt hữu ích với:

[
K=1,2,4.
]

---

# 11. Stage 4 — Ước lượng coalition gain

Đây là heart của proposed.

Ta có current optimized adversarial candidate:

[
x_S=x+\delta_S.
]

Ta muốn biết với mỗi:

[
j\notin S,
]

pixel đó có hợp với current coalition không.

Exact:

[
\Delta(j\mid S)
===============

F(S\cup{j})-F(S).
]

Nhưng tính exact cho mọi (j) cần optimize lại hàng chục lần.

Ta cần proxy.

## Phiên bản V1: conditional box-aware gradient

Tính gradient tại **current coalition solution**:

[
g^S=
\nabla_x J(x_S,y).
]

Sau đó:

[
\widehat{\Delta}_{1}(j\mid S)
=============================

\max_{v_j}
(g^S_j)^\top v_j.
]

Điểm rất quan trọng:

Gradient này **không tính tại clean x**.

Nó tính tại:

[
x+\delta_S.
]

Vì vậy score của (j) phụ thuộc vào coalition hiện tại.

Đây đã là một form rất rẻ của interaction awareness.

Hai supports khác nhau:

[
S_1,S_2
]

cho hai gradients khác nhau:

[
g^{S_1}\neq g^{S_2}.
]

Do đó:

[
\widehat{\Delta}(j\mid S_1)
\neq
\widehat{\Delta}(j\mid S_2).
]

---

# 12. Stage 4 nâng cao — explicit interaction correction

Nếu chỉ conditional gradient vẫn chưa đủ novelty/strength, thêm second-order correction.

Taylor expansion:

[
J(x+\delta)
\approx
J(x)
+
g^\top\delta
+
\frac12\delta^\top H\delta.
]

Với current support (S) và candidate (j), interaction giữa candidate và coalition xấp xỉ:

[
\boxed{
I(j,S)
\approx
\delta_j^\top
H_{jS}
\delta_S
}
]

Trong đó:

[
H_{jS}
======

\frac{\partial^2J}
{\partial x_j\partial x_S}.
]

Sau đó:

[
\widehat{\Delta}(j\mid S)
=========================

A_j^S
+
\lambda_I I(j,S).
]

Trong implementation không nên build Hessian đầy đủ.

Có thể dùng Hessian-vector product:

[
Hv=
\nabla_x
\left(
\nabla_xJ^\top v
\right).
]

Với:

[
v=\delta_S.
]

Ta lấy components ngoài support để estimate:

[
(H\delta_S)_j.
]

Đây sẽ cho một approximation của:

> pixel (j) tương tác với perturbation hiện tại như thế nào.

Nhưng mình sẽ **không bắt đầu V2 bằng Hessian**.

Hãy làm conditional-gain V1 trước, benchmark, rồi mới quyết định second-order có đáng compute không.

---

# 13. Stage 5 — Tìm pixel redundant trong support

Không chỉ tìm pixel ngoài support mạnh.

Ta cần biết:

> Pixel nào đang chiếm budget nhưng ít đóng góp cho coalition?

Exact removal cost:

[
R(i\mid S)
==========

F(S)-F(S\setminus{i}).
]

Nếu:

[
R(i\mid S)\approx0,
]

pixel (i) gần như redundant.

Nếu:

[
R(i\mid S)<0,
]

bỏ nó thậm chí còn làm objective tốt hơn.

Exact calculation lại tốn.

Cheap approximation:

[
\boxed{
\widehat R_i
============

g_i^\top\delta_i
}
]

tại current adversarial point.

Hoặc tốt hơn một chút:

set pixel (i) về clean value và forward một lần:

[
\tilde{x}^{(-i)}
================

x_S-\delta_i.
]

Rồi:

[
R_i^{exact-local}
=================

J(x_S)-J(\tilde{x}^{(-i)}).
]

Vì support chỉ có K pixels, với K≤64 thì evaluate active pixels tương đối khả thi.

Mình sẽ dùng:

* gradient estimate để pre-rank;
* exact local removal cho vài weakest pixels.

---

# 14. Stage 6 — Support exchange

Bây giờ ta có:

Outside candidates:

[
\widehat{\Delta}(j\mid S)
]

và active removal cost:

[
\widehat R(i\mid S).
]

Chọn:

[
i^*
===

\arg\min_{i\in S}\widehat R(i\mid S),
]

[
j^*
===

\arg\max_{j\notin S}\widehat\Delta(j\mid S).
]

Proposal:

[
S'
==

S\setminus{i^*}
\cup
{j^*}.
]

Đây là key difference với Top-K attack.

Top-K hỏi:

[
\text{pixel nào individually có score cao?}
]

CASA hỏi:

[
\boxed{
\text{pixel nào bên ngoài bổ sung tốt nhất cho coalition,
và pixel nào bên trong redundant nhất?}
}
]

---

# 15. Không accept exchange chỉ vì approximation nói tốt

Proxy có thể sai.

Do đó sau proposal:

[
S\rightarrow S',
]

ta copy perturbations của retained pixels:

[
\delta_{S\cap S'}
]

và initialize new pixel (j^*).

Sau đó chạy vài inner optimization steps:

[
\delta_{S'}^*
\leftarrow
\operatorname{Optimize}(S').
]

Tính objective thật:

[
F_{\text{old}}
==============

J(x+\delta_S^*,y),
]

[
F_{\text{new}}
==============

J(x+\delta_{S'}^*,y).
]

Accept nếu:

[
F_{\text{new}}

>

F_{\text{old}}+\epsilon.
]

Nếu không:

[
S'\rightarrow S.
]

Đây tạo một property đẹp:

> **Mỗi accepted support exchange không làm giảm evaluated attack objective.**

Nó không guarantee global optimum.

Nhưng algorithm story rất sạch.

---

# 16. Có thể swap nhiều hơn một pixel

Single swap:

[
1\text{-out}/1\text{-in}
]

là version ổn định nhất.

Sau đó có thể thử:

[
2\text{-out}/2\text{-in}.
]

Ví dụ synergy đôi khi yêu cầu hai individually weak pixels phải cùng vào support.

Single swap có thể không vượt local optimum.

Giả sử:

[
S={A,B}.
]

Candidate C một mình không đủ tốt để thay A.

Candidate D một mình cũng không đủ.

Nhưng:

[
{C,D}
]

cực mạnh.

Single exchange sẽ không tìm được.

Do đó V2 advanced có thể có **pair proposal**:

[
(i_1,i_2)
\rightarrow
(j_1,j_2).
]

Nhưng computational cost tăng đáng kể.

Mình sẽ để pair-swap làm ablation/extension, không phải initial core.

---

# 17. Một giải pháp hay hơn để khám phá true synergy

Để tìm pair coalition mà không brute-force toàn ảnh:

Đầu tiên shortlist:

[
C_M,\quad M=32.
]

Sau đó chỉ xét candidate pairs trong pool:

[
{32\choose2}=496.
]

Có thể đánh giá approximate pair gain:

[
\widehat I(i,j)
]

rồi lấy Top-P pair, ví dụ 8 hoặc 16 pair, mới exact optimize.

Như vậy ta có một **exploration step** để thoát khỏi single-pixel local search.

CASA có thể xen kẽ:

[
\text{single exchange}
\rightarrow
\text{single exchange}
\rightarrow
\text{pair exploration}
\rightarrow
...
]

Điều này rất hợp với idea coalition.

---

# 18. Khi nào dừng support refinement?

Outer loop dừng khi một trong các điều kiện xảy ra:

[
J(x_{\mathrm{adv}},y)>0
]

và nếu mục tiêu chỉ là success-first thì chuyển sang minimization.

Hoặc:

[
N_{\text{no-improve}}\ge P.
]

Hoặc hết:

[
T_{\text{outer}}.
]

Hoặc query/backward budget.

Để benchmark fairness, algorithm nên expose:

```text
outer_steps
inner_steps
candidate_pool
max_swaps
pair_search_every
forward_evals
backward_evals
```

---

# 19. Sau khi attack thành công: không dừng ở K

Giả sử:

[
|S|=16
]

và attack success.

Nhưng có thể actual minimum support chỉ là 7.

Current SFA dùng greedy pruning: thử set một modified pixel về zero, nếu vẫn success thì bỏ.

Proposed mới dùng:

[
\boxed{\text{Drop-and-Repair}}
]

---

# 20. Drop-and-Repair

Ta rank active pixels theo removal importance:

[
R(i\mid S).
]

Chọn weakest:

[
i^*=\arg\min R(i\mid S).
]

Drop:

[
S'=S\setminus{i^*}.
]

Nếu trực tiếp vẫn adversarial:

[
J(x+\delta_{S'},y)>0,
]

giữ removal.

Nếu bị mất success, **không revert ngay**.

Ta re-optimize remaining RGB values:

[
\delta_{S'}\leftarrow
\operatorname{Optimize}(S').
]

Nếu sau repair:

[
J(x+\delta_{S'},y)>0,
]

ta giảm được:

[
K\rightarrow K-1.
]

Nếu vẫn fail, có thể chạy vài support exchanges ở budget mới:

[
|S'|=K-1.
]

Tức ta hỏi:

> Có support khác với K-1 pixels vẫn attack được không?

Nếu có, tiếp tục.

---

# 21. Điều này khác greedy pruning rất nhiều

Giả sử current adversarial:

[
S={A,B,C,D}.
]

Remove D trực tiếp:

[
J<0.
]

Greedy pruning kết luận:

> D cần thiết.

Nhưng có thể sau khi remove D và optimize lại A,B,C:

[
J>0.
]

Hoặc swap:

[
C\rightarrow E
]

thì:

[
{A,B,E}
]

success.

Greedy pruning không tìm được.

Drop-and-Repair thì có thể.

---

# 22. Budgeted mode và minimal mode

CASA nên có hai operation modes.

### Budgeted CASA

Input:

[
K.
]

Output best adversarial under:

[
L_0\le K.
]

Dùng để benchmark trực tiếp:

[
ASR@K.
]

Đây là primary mode.

### Minimal CASA

Có thể bắt đầu từ:

[
K_{\max}
]

rồi:

[
K_{\max}\rightarrow K_{\max}-1
\rightarrow...
]

bằng Drop-and-Repair.

Output:

[
L_0^{\min}
]

hoặc near-minimal L0.

Như vậy cùng architecture có thể cạnh tranh ở cả hai evaluation styles.

---

# 23. Feature guidance của SFA cũ đi đâu?

Không nên xóa ngay.

Current SFA có:

[
L=
L_{\mathrm{cls}}
+
\lambda_fL_{\mathrm{feature}}.
]

Trong CASA, feature information nên trở thành **optional coalition prior**, không phải core objective.

Ví dụ:

[
\widehat{\Delta}_{total}(j\mid S)
=================================

\widehat{\Delta}*{margin}(j\mid S)
+
\lambda_f
\widehat{\Delta}*{feature}(j\mid S).
]

Nhưng default:

[
\lambda_f=0.
]

Lý do là paper cần chứng minh:

> coalition-aware support optimization tự thân đã mạnh.

Sau đó mới thêm feature.

Nếu feature không cải thiện rõ → bỏ khỏi final proposed.

---

# 24. CPA cũ cũng không cần mất

CPA hiện dùng neighboring gradient alignment như proxy cho cooperation.

Nó sẽ trở thành một rất tốt ablation:

[
\text{CPA heuristic interaction}
]

vs

[
\text{CASA conditional coalition gain}.
]

Nếu CASA thắng rõ, ta có một story đẹp:

> Gradient alignment không phải là proxy đủ tốt cho true coalition utility.

---

# 25. Pseudocode cấp cao

Conceptual algorithm:

```text
Input:
    model f
    clean image x
    label y
    sparse budget K

Compute clean margin J(x,y)

1. Candidate initialization
    g ← ∇x J(x,y)
    A ← box-aware pixel gain(g, x)
    S ← TopK(A)

2. Optimize values on S
    δS ← optimize RGB values under fixed support S
    best ← J(x + δS, y)

3. Coalition refinement
    repeat:
        gS ← ∇x J(x + δS, y)

        candidate_gain[j] ← conditional box-aware gain
                              for j ∉ S

        removal_cost[i] ← contribution estimate
                           for i ∈ S

        i* ← weakest active pixel
        j* ← best complementary inactive pixel

        S' ← S - {i*} + {j*}

        warm-start δS'
        optimize δS'

        if J(x + δS', y) > best:
            S ← S'
            δS ← δS'
            best ← new objective
        else:
            reject

        periodically run pair exploration

    until success / no improvement / budget exhausted

4. If successful:
    run Drop-and-Repair
        try K → K-1
        reoptimize
        optionally support-exchange
        keep smaller support if still successful

5. Return:
    x_adv
    final support
    final L0
    margin
    forward/backward/query counts
```

---

# 26. Flow toàn bộ method

Có thể hình dung:

```text
                    CLEAN IMAGE
                        │
                        ▼
                decision margin
                        │
                        ▼
             box-aware pixel gain
                        │
                        ▼
               candidate shortlist
                        │
                        ▼
               initial support S
                        │
                        ▼
            optimize RGB values on S
                        │
                        ▼
              current coalition state
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
      outside conditional     active pixel
       coalition gain         redundancy
              │                   │
              └─────────┬─────────┘
                        ▼
               support exchange
             weak OUT ↔ useful IN
                        │
                        ▼
                 re-optimize RGB
                        │
                        ▼
                  exact evaluate
                ┌───────┴────────┐
              better           worse
                │                │
             accept            reject
                │                │
                └──────┬─────────┘
                       ▼
                adversarial?
                  │          │
                 no         yes
                  │          │
             continue   Drop-and-Repair
                             │
                             ▼
                     smaller support?
                       │           │
                      yes          no
                       │           │
                    repeat        stop
                             │
                             ▼
                         OUTPUT
```

---

# 27. Vì sao method này khác Sparse-PGD?

Sparse-PGD cũng jointly optimize sparse mask và perturbation.

Điểm proposed cần nhấn mạnh không phải:

> “chúng tôi cũng optimize support.”

Mà là:

[
\boxed{
\text{Support elements are evaluated conditionally as a coalition}
}
]

CASA explicitly có:

[
\Delta(j\mid S)
]

và:

[
R(i\mid S).
]

Sparse support không còn là một collection của independently scored pixels.

Nó là một **set whose members change utility depending on the rest of the set**.

Đây là conceptual contribution.

---

# 28. Vì sao khác GSE?

GSE quan tâm group-wise / structured sparsity.

CASA không yêu cầu:

[
i,j\text{ phải spatially gần nhau}.
]

Hai pixels:

[
i=(2,4),
\qquad
j=(27,29)
]

vẫn có thể thuộc cùng coalition nếu:

[
\Delta(j\mid{i})
]

cao.

Do đó CASA grouping là:

[
\boxed{\text{functional/adversarial grouping}}
]

không phải:

[
\text{spatial grouping}.
]

---

# 29. Vì sao khác current CPA?

CPA giả định:

[
\text{gradient alignment}
\Rightarrow
\text{cooperation}.
]

CASA hỏi trực tiếp:

[
\boxed{
\text{Candidate làm objective tăng thêm bao nhiêu
khi đặt vào coalition hiện tại?}
}
]

CPA là static/local proxy.

CASA là conditional support utility.

Đây là bước tiến logic rất rõ từ SFA-v1 sang v2.

---

# 30. Một ví dụ đầy đủ

Giả sử:

[
K=3.
]

Candidate individual scores:

[
A=1.0,\quad
B=0.9,\quad
C=0.8,\quad
D=0.6,\quad
E=0.5.
]

Top-K attack chọn:

[
S_0={A,B,C}.
]

Sau optimization:

[
F(S_0)=1.1.
]

Giả sử chưa success vì cần:

[
J>1.5.
]

CASA đo redundancy:

[
R(A)=0.7,\quad
R(B)=0.05,\quad
R(C)=0.4.
]

B gần như redundant.

Outside conditional gains:

[
\Delta(D\mid S_0)=0.2,
]

[
\Delta(E\mid S_0)=1.0.
]

D có individual score lớn hơn E:

[
0.6>0.5,
]

nhưng E **complements coalition tốt hơn**.

CASA proposal:

[
S_1=
{A,C,E}.
]

Optimize lại:

[
F(S_1)=2.0.
]

Attack success.

Sau đó Drop-and-Repair.

Bỏ C:

[
{A,E}
]

ban đầu:

[
J=1.2.
]

Nhưng sau reoptimization:

[
F({A,E})=1.7.
]

Vẫn success.

Final:

[
L_0=2
]

dù attack chạy với:

[
K=3.
]

Đó là toàn bộ philosophy của method.

---

# 31. Computational complexity

Giả sử:

[
M=32
]

candidate pool,

[
K=16.
]

Mỗi outer step chỉ cần:

* 1 backward để conditional score;
* inner optimization khoảng 5–10 backwards;
* vài exact forward cho removal candidates;
* một proposed swap;
* periodic pair exploration.

Nó sẽ đắt hơn vanilla PGD0/sPGD.

Nhưng có thể rẻ hơn brute-force interaction cực nhiều.

Do đó benchmark phải report:

[
ASR,\quad
L_0,\quad
forward,\quad
backward,\quad
runtime.
]

Method không cần là fastest nếu nó đạt better:

[
ASR@smallK
]

hoặc lower:

[
L_0.
]

---

# 32. Hyperparameters ban đầu mình đề xuất

Cho CIFAR-10:

[
K\in{1,2,4,8,16,32,64}.
]

Candidate pool:

[
M=\min(64,4K)
]

nhưng ít nhất khoảng 16.

Inner optimization:

[
T_{\text{inner}}=10
]

cho initialization,

sau mỗi swap:

[
T_{\text{repair}}=5.
]

Outer exchanges:

[
T_{\text{outer}}=20.
]

Pair exploration:

mỗi 4–5 accepted/rejected swaps.

Loss:

[
\boxed{\text{margin loss}}
]

không phải CE.

Feature weight ban đầu:

[
\lambda_f=0.
]

Interaction second-order:

[
\lambda_H=0
]

ở version đầu.

Hãy chứng minh conditional support search trước.

---

# 33. Ablation bắt buộc

Method phải được xây theo cách mỗi contribution bật/tắt được.

Base:

[
\text{Top-K box gain + fixed-support RGB optimization}.
]

Sau đó:

[
+\text{conditional rescoring}.
]

Sau đó:

[
+\text{support exchange}.
]

Sau đó:

[
+\text{pair exploration}.
]

Sau đó:

[
+\text{Drop-and-Repair}.
]

Cuối cùng optional:

[
+\text{second-order interaction},
]

[
+\text{feature prior}.
]

Nếu final method thắng nhưng ablation không chứng minh coalition mechanism tạo gain thì contribution sẽ yếu.

---

# 34. Metrics riêng để chứng minh idea

Ngoài ASR, cần log các quantity liên quan chính hypothesis.

**Support turnover**

[
T_t
===

1-
\frac{|S_t\cap S_{t-1}|}
{|S_t\cup S_{t-1}|}.
]

**Accepted swap rate**

[
\frac{#accepted}
{#proposals}.
]

**Average objective gain per accepted swap**

[
\Delta J_{\text{swap}}.
]

**Redundancy removed**

so sánh removal cost của removed pixels với random pixels.

**Conditional-vs-independent disagreement**

Bao nhiêu % candidate:

[
\arg\max A_j
\neq
\arg\max\Delta(j\mid S).
]

Nếu tỷ lệ này cao và conditional candidate thường tốt hơn, ta có evidence trực tiếp cho research hypothesis.

---

# 35. Figure quan trọng nhất cho paper

Mình rất muốn có một figure dạng:

```text
Individual ranking:

A = 0.9
B = 0.8
C = 0.5

Top-2 → {A,B}
Joint gain = 1.0


Coalition-aware:

A = 0.9
C = 0.5

Conditional gain:
C | {A} = +1.2

Selected → {A,C}
Joint gain = 2.1
```

Một figure như vậy giải thích toàn paper trong vài giây.

---

# 36. Research hypothesis của paper

Có thể formalize thành:

> **H1:** Individual pixel importance is insufficient for constructing optimal sparse adversarial supports because candidate utility depends on the current support.

> **H2:** Explicitly replacing redundant active pixels with complementary inactive pixels improves attack success under fixed (L_0) budgets.

> **H3:** Reoptimization after support reduction yields smaller adversarial supports than greedy perturbation pruning.

Nếu experiment chứng minh đủ cả 3, paper story rất coherent.

---

# 37. Contribution cuối cùng nên được kể như thế nào

Không nên kể:

> Chúng tôi đề xuất feature loss + interaction + pruning.

Nên kể:

> Existing sparse attacks typically optimize sparsity while treating support selection through independent importance scores, continuous masks, projections, or implicit global search. We instead model sparse support as a coalition whose pixel utilities depend on the other selected pixels.

Sau đó ba mechanisms nối nhau:

[
\boxed{
\text{conditional coalition gain}
}
]

[
\boxed{
\text{redundancy-aware support exchange}
}
]

[
\boxed{
\text{drop-and-repair support minimization}
}
]

Tất cả đều phục vụ cùng một thesis.

---

# 38. Cách tích hợp vào AA_2

Mình sẽ **không sửa `SparseFeatureAttack` hiện tại**.

Giữ:

```text
src/aa/attacks/proposed.py
```

làm SFA-v1.

Tạo mới:

```text
src/aa/attacks/casa.py
```

và các component:

```text
CoalitionSparseAttack
BoxAwareScorer
SupportOptimizer
SupportExchange
DropRepair
```

Registry có thể tạm thêm:

```text
ours_v1 -> SparseFeatureAttack
ours_v2 -> CoalitionSparseAttack
```

Sau khi v2 ổn mới đổi:

```text
ours -> CASA
```

Như vậy benchmark được:

[
\text{SFA-v1}
\quad vs\quad
\text{CASA-v2}
]

trực tiếp trong cùng infrastructure.

---

# 39. Roadmap triển khai mình khuyên

Version đầu tiên **không làm Hessian, không Shapley, không feature loss**.

Làm một CASA-lite gồm:

[
\boxed{
\text{margin}
+
\text{box-aware candidate score}
+
\text{conditional rescoring}
+
\text{1-out/1-in support exchange}
+
\text{fixed-support RGB optimization}
+
\text{Drop-and-Repair}
}
]

Nếu CASA-lite đã thắng SFA / PGD0 / sPGD đáng kể ở:

[
K=4,8,16,
]

thì hypothesis có tín hiệu mạnh.

Sau đó mới thêm true pair interaction hoặc Hessian correction.

Đây là cách giảm rủi ro nghiên cứu: **test central idea trước, complexity sau**.

---

## Tóm lại

Proposed mới không còn nhìn sparse perturbation như:

[
\boxed{
\text{K individually strongest pixels}
}
]

mà như:

[
\boxed{
S^*
===

\arg\max_{|S|\le K}F(S)
}
]

trong đó utility của một pixel phụ thuộc vào coalition:

[
\boxed{
\Delta(j\mid S)
===============

F(S\cup{j})-F(S).
}
]

Algorithm giải gần đúng bài toán này bằng:

[
\boxed{
\text{Cheap candidate screening}
\rightarrow
\text{fixed-support value optimization}
\rightarrow
\text{conditional coalition scoring}
\rightarrow
\text{redundancy-aware exchange}
\rightarrow
\text{exact accept/reject}
\rightarrow
\text{Drop-and-Repair}
}
]

Điểm mình thích nhất ở hướng này là **mọi component đều xuất phát từ cùng một vấn đề nghiên cứu**: sparse support là một coalition, không phải Top-K collection. Nếu triển khai tốt, đây có research identity rõ hơn đáng kể so với SFA hiện tại.
