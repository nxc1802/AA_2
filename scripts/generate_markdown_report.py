import json

def generate_report():
    data = json.load(open('result/paper_cifar10_results.json'))
    meta = data.get('metadata', {})
    cfg = meta.get('config', {})
    ds_cfg = cfg.get('dataset', {})
    model_cfg = cfg.get('model', {})
    results = data.get('results', {})
    repro = meta.get('reproducibility', {})

    md = []
    md.append('# CIFAR-10 Adversarial Attack Benchmark Results Report\n')
    md.append('This document provides a comprehensive, detailed report of the 1,000-sample CIFAR-10 attack benchmark executed on Kaggle with strict paper validation.\n')

    md.append('## 1. Environment & Experiment Metadata\n')
    md.append('| Parameter | Value |')
    md.append('| :--- | :--- |')
    md.append(f'| **Dataset** | {ds_cfg.get("name", "cifar10").upper()} ({ds_cfg.get("samples")} samples, batch size {ds_cfg.get("batch_size")}) |')
    md.append(f'| **Model Architecture** | ResNet-18 (CIFAR variant) |')
    md.append(f'| **Model Checkpoint** | `{model_cfg.get("checkpoint")}` |')
    md.append(f'| **Checkpoint SHA256** | `{meta.get("model_checkpoint_sha256")}` |')
    md.append(f'| **Expected Clean Accuracy** | {model_cfg.get("expected_clean_acc")}% |')
    md.append(f'| **Strict Validation Mode** | `{meta.get("strict_mode")}` |')
    md.append(f'| **Device** | `{meta.get("device")}` |')
    md.append(f'| **Random Seed** | `{cfg.get("seed")}` |')
    md.append(f'| **Git Commit** | `{repro.get("git_commit")}` |')
    md.append(f'| **Python Version** | `{repro.get("python_version")}` |')
    md.append(f'| **PyTorch Version** | `{repro.get("pytorch_version")}` |\n')

    md.append('---\n')
    md.append('## 2. Main Benchmark Summary Tables\n')

    md.append('### Table 1: Dense Reference Attacks ($L_{\infty} / L_2$ Baseline)\n')
    md.append('| Attack | Clean Acc (%) | Robust Acc (%) | ASR (%) | Forward Evals | Backward Evals | Runtime (s) |')
    md.append('| :--- | :---: | :---: | :---: | :---: | :---: | :---: |')
    for atk_name in ['fgsm', 'bim', 'pgd']:
        if atk_name in results and 'dense' in results[atk_name]:
            d = results[atk_name]['dense']
            md.append(f'| **{atk_name.upper()}** | {d["clean_accuracy"]:.2f}% | {d["robust_accuracy"]:.2f}% | {d["asr"]:.2f}% | {d.get("total_forward_evals", 0)} | {d.get("total_backward_evals", 0)} | {d["runtime_seconds"]:.1f}s |')
    md.append('\n')

    k_vals = cfg.get('benchmark', {}).get('k_values', [1, 2, 4, 8, 16, 32, 64])

    md.append('### Table 2: Attack Success Rate ($ASR@K$) Comparison (%)\n')
    md.append('| Attack | ' + ' | '.join([f'K={k}' for k in k_vals]) + ' |')
    md.append('| :--- | ' + ' | '.join([':---:' for _ in k_vals]) + ' |')

    sparse_atks = [a for a in results if a not in ['fgsm', 'bim', 'pgd']]
    for atk_name in sparse_atks:
        display_name = 'Ours (SparseFeatureAttack)' if atk_name == 'ours' else atk_name.upper()
        row = [f'**{display_name}**']
        for k in k_vals:
            k_key = f'k_{k}'
            k_data = results[atk_name].get(k_key, {})
            asr = k_data.get('asr', 0.0)
            row.append(f'{asr:.2f}%')
        md.append('| ' + ' | '.join(row) + ' |')
    md.append('\n')

    md.append('### Table 3: Conditional Robust Accuracy ($CRA@K$) Comparison (%)\n')
    md.append('| Attack | ' + ' | '.join([f'K={k}' for k in k_vals]) + ' |')
    md.append('| :--- | ' + ' | '.join([':---:' for _ in k_vals]) + ' |')
    for atk_name in sparse_atks:
        display_name = 'Ours (SparseFeatureAttack)' if atk_name == 'ours' else atk_name.upper()
        row = [f'**{display_name}**']
        for k in k_vals:
            k_key = f'k_{k}'
            k_data = results[atk_name].get(k_key, {})
            cra = k_data.get('conditional_robust_accuracy', 0.0)
            row.append(f'{cra:.2f}%')
        md.append('| ' + ' | '.join(row) + ' |')
    md.append('\n')

    md.append('### Table 4: Runtime per Attack & Budget (Seconds)\n')
    md.append('| Attack | ' + ' | '.join([f'K={k}' for k in k_vals]) + ' | Total Time |')
    md.append('| :--- | ' + ' | '.join([':---:' for _ in k_vals]) + ' | :---: |')
    for atk_name in sparse_atks:
        display_name = 'Ours (SparseFeatureAttack)' if atk_name == 'ours' else atk_name.upper()
        row = [f'**{display_name}**']
        tot_t = 0.0
        for k in k_vals:
            k_key = f'k_{k}'
            k_data = results[atk_name].get(k_key, {})
            rt = k_data.get('runtime_seconds', 0.0)
            if atk_name in ['cornersearch', 'sparsefool', 'sigma_zero', 'gse']:
                tot_t = rt
            else:
                tot_t += rt
            row.append(f'{rt:.1f}s')
        row.append(f'{tot_t:.1f}s')
        md.append('| ' + ' | '.join(row) + ' |')
    md.append('\n')

    md.append('---\n')
    md.append('## 3. Per-Attack Detailed Breakdown\n')

    for atk_name, atk_data in results.items():
        display_name = 'Ours (SparseFeatureAttack)' if atk_name == 'ours' else atk_name.upper()
        md.append(f'### {display_name}\n')
        if 'dense' in atk_data:
            d = atk_data['dense']
            m = d.get('metrics', {})
            md.append(f'- **Clean Accuracy**: {d.get("clean_accuracy"):.2f}% ({d.get("clean_correct_count")}/{d.get("total_samples")})')
            md.append(f'- **Robust Accuracy**: {d.get("robust_accuracy"):.2f}%')
            md.append(f'- **ASR**: {d.get("asr"):.2f}%')
            md.append(f'- **Mean L0**: {m.get("succ_l0_mean", 0):.2f} (Median: {m.get("succ_l0_median", 0)})')
            md.append(f'- **Mean L2**: {m.get("succ_l2_mean", 0):.4f}')
            md.append(f'- **Mean Linf**: {m.get("succ_linf_mean", 0):.4f}')
            md.append(f'- **Mean PSNR**: {m.get("succ_psnr_mean", 0):.2f} dB')
            md.append(f'- **Mean SSIM**: {m.get("succ_ssim_mean", 0):.4f}')
            md.append(f'- **Runtime**: {d.get("runtime_seconds"):.2f}s\n')
        else:
            md.append('| K | Success Count | ASR (%) | Cond Robust Acc (%) | Mean L0 | Median L0 | Mean L2 | Mean Linf | Mean PSNR (dB) | Mean SSIM | Queries/img | Runtime (s) |')
            md.append('| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |')
            for k in k_vals:
                k_key = f'k_{k}'
                if k_key in atk_data:
                    kd = atk_data[k_key]
                    m = kd.get('metrics', {})
                    succ_cnt = kd.get('success_count', 0)
                    asr = kd.get('asr', 0.0)
                    cra = kd.get('conditional_robust_accuracy', 0.0)
                    l0_m = m.get('succ_l0_mean', 0.0)
                    l0_med = m.get('succ_l0_median', 0)
                    l2_m = m.get('succ_l2_mean', 0.0)
                    linf_m = m.get('succ_linf_mean', 0.0)
                    psnr_m = m.get('succ_psnr_mean', 0.0)
                    ssim_m = m.get('succ_ssim_mean', 0.0)
                    q = kd.get('queries_per_image', 0.0)
                    rt = kd.get('runtime_seconds', 0.0)
                    md.append(f'| {k} | {succ_cnt} | {asr:.2f}% | {cra:.2f}% | {l0_m:.2f} | {l0_med} | {l2_m:.4f} | {linf_m:.4f} | {psnr_m:.2f} | {ssim_m:.4f} | {q:.1f} | {rt:.1f}s |')
            md.append('\n')

    with open('docs/benchmark_results_cifar10.md', 'w') as f:
        f.write('\n'.join(md))

    print('Successfully generated docs/benchmark_results_cifar10.md!')

if __name__ == '__main__':
    generate_report()
