Mỗi clean model chỉ training đúng một lần trên split 40k/10k.

Sử dụng CIFAR-10 làm benchmark chính và CIFAR-100 làm benchmark mở rộng. Mỗi dataset được tải đầy đủ, gồm 50.000 training images và 10.000 test images, kích thước đầu vào 3\times32\times32, không resize lên 224\times224.

Tập training chính thức được chia cố định thành 40.000 train và 10.000 validation, sử dụng một random seed cố định và stratified split để giữ tỷ lệ lớp. Tập 10.000 test images được giữ riêng và chỉ dùng cho báo cáo kết quả cuối. Không thực hiện bước train lại trên toàn bộ 50.000 ảnh sau khi chọn cấu hình.

Sử dụng hai clean backbones: CIFAR-adapted ResNet-18 và WideResNet-28-10. ResNet-18 dùng conv1 = 3×3, stride=1, bỏ initial max-pooling, giữ residual configuration [2,2,2,2] và thay classifier thành số lớp tương ứng: 10 classes cho CIFAR-10 và 100 classes cho CIFAR-100. WRN-28-10 sử dụng kiến trúc chuẩn dành cho CIFAR. Các clean model được train from scratch, không dùng ImageNet pretrained weights.

Mỗi model-dataset pair chỉ được training một lần duy nhất trên 40.000 training samples. Validation set 10.000 ảnh được dùng để theo dõi loss và accuracy trong quá trình training, đồng thời lưu checkpoint có validation loss thấp nhất và tự động đẩy lên Hugging Face repository (`Cuong2004/AA`). Cấu hình training: SGD, momentum=0.9, weight_decay=5e-4, batch_size=256 cho train_loader và 512 cho val/test_loader, tối đa 200 epochs, Early Stopping theo dõi Validation Loss với Patience=20 epochs, initial learning rate=0.1, cosine annealing, RandomCrop(32, padding=4) và RandomHorizontalFlip. Validation, test và attack không sử dụng augmentation ngẫu nhiên.

Bộ clean checkpoints gồm:

* CIFAR-10 ResNet-18.
* CIFAR-10 WRN-28-10.
* CIFAR-100 ResNet-18.
* CIFAR-100 WRN-28-10.

Nếu muốn giảm chi phí ban đầu, ưu tiên hoàn thành CIFAR-10 ResNet-18, các checkpoint chỉ được chạy sau khi attack pipeline đã ổn định.

Phần attack gồm các method đã có sẵn, cùng proposed sparse attack. Các attack được chạy trên cùng test set, cùng sample indices, cùng threat model và cùng perturbation budget khi so sánh trực tiếp.

Báo cáo Clean Accuracy trên toàn bộ 10.000 test images, Robust Accuracy trên toàn bộ test set và conditional ASR trên tập con các ảnh được clean model phân loại đúng. Việc lọc clean-correct chỉ áp dụng khi tính ASR; attack vẫn có thể được sinh và robust accuracy vẫn được tính trên toàn bộ test set.

Phần preprocessing defense đã có sẵn. Mỗi preprocessing method được đặt trước classifier và đánh giá cả clean accuracy sau preprocessing lẫn robust accuracy dưới attack để đo trade-off giữa khả năng phòng thủ và suy giảm chất lượng ảnh.

Phần learning-based defense sử dụng hai model từ RobustBench checkpoint, ưu tiên:
- Robust ResNet18, PGD Adversarial Training
- Robust WRN28-10, TRADES
Khi báo cáo phải ghi rõ backbone, training method, dataset, threat model, perturbation budget và nguồn checkpoint. Proposed attack và các baseline chính được chạy lại trên hai robust models để đánh giá cross-defense và cross-threat-model robustness.

Các metric đánh giá cuối cùng được chia thành bốn nhóm:

Hiệu quả tấn công: Clean Accuracy, Robust Accuracy, Accuracy Drop, conditional Attack Success Rate (ASR) và success rate theo từng perturbation budget.

Độ thưa và cường độ perturbation: số pixel hoặc vị trí spatial bị thay đổi (L_0), tỷ lệ pixel bị thay đổi (L_0/(H\times W)), (L_1), (L_2), (L_\infty), giá trị perturbation trung bình trên các pixel bị thay đổi và ngân sách sparse thực tế được sử dụng khi attack thành công.

Chất lượng và độ tương đồng ảnh: Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), Multi-Scale SSIM (MS-SSIM) và Learned Perceptual Image Patch Similarity (LPIPS). Có thể bổ sung Mean Squared Error (MSE) hoặc Mean Absolute Error (MAE) để hỗ trợ phân tích định lượng, nhưng không sử dụng chúng làm metric cảm nhận chính. PSNR và SSIM đo mức sai khác pixel và cấu trúc; LPIPS đo khoảng cách perceptual dựa trên deep features. Với ảnh CIFAR-10 kích thước (32\times32), cần báo cáo trung bình, độ lệch chuẩn và phân phối metric trên toàn bộ adversarial examples thành công.

Hiệu quả tính toán và khả năng phân tích: runtime trung bình trên mỗi ảnh, số lần forward/backward, query count đối với black-box attack, số iteration đến khi thành công, memory usage nếu cần, cùng visualization gồm clean image, adversarial image, perturbation map, binary sparse mask và ảnh phóng đại phần perturbation.

Các metric chất lượng ảnh phải được tính giữa ảnh clean (x) và ảnh adversarial (x_{\mathrm{adv}}) trong cùng miền pixel ([0,1]), trước normalization và trước preprocessing defense. Kết quả nên được báo cáo riêng cho toàn bộ mẫu và cho tập con attack thành công. Khi so sánh các attack, cần sử dụng cùng tập sample indices và cùng ngân sách (L_0).

Đánh giá proposed sparse attack trên nhiều mức sparse budget K thay vì chỉ một giá trị cố định. Với CIFAR-10, sử dụng các mức K=\{1 2 4 8 16 32 64 128\} và áp dụng đồng nhất cho tất cả các sparse attack có hỗ trợ giới hạn ngân sách như JSMA, SparseFool, CornerSearch, OnePixel, Pixle và proposed method. Đối với các dense attack (FGSM, BIM, PGD), sử dụng cấu hình chuẩn của từng phương pháp và không đưa vào so sánh trực tiếp theo K. Thực hiện ablation theo từng mức K, xây dựng ASR–K Curve, Robust Accuracy–K Curve và đánh giá đồng thời PSNR, SSIM, LPIPS, runtime theo từng ngân sách. Mục tiêu là chứng minh proposed method đạt Attack Success Rate cao hơn, chất lượng ảnh tốt hơn và chi phí tính toán thấp hơn dưới cùng một sparse budget, đồng thời phân tích sự đánh đổi (trade-off) giữa mức độ thưa (sparsity) và hiệu quả tấn công.