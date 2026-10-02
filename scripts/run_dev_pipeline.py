#!/usr/bin/env python3
import os
import sys
import time
import json
import subprocess
import traceback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
RESULT_DIR = os.path.join(PROJECT_ROOT, "result")
LOG_PATH = os.path.join(RESULT_DIR, "active_run.log")
FLAG_PATH = os.path.join(RESULT_DIR, "pipeline_done.flag")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

def log(msg):
    t_str = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{t_str} {msg}\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line)

def send_toast(msg, kind="info"):
    try:
        import marimo as mo
        mo.status.toast(msg, kind=kind)
    except Exception:
        pass

def run_command_stream(cmd, stage_name):
    log(f"=== [START STAGE: {stage_name}] ===")
    log(f"Command: {cmd}")
    t0 = time.time()
    send_toast(f"🚀 Started: {stage_name}", kind="info")
    
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    last_heartbeat = time.time()
    for line in iter(proc.stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()
        with open("/marimo/result/active_run.log", "a") as f:
            f.write(line)
            
        # Heartbeat every 30 seconds to keep connection vibrant
        if time.time() - last_heartbeat > 30:
            last_heartbeat = time.time()
            log(f"Heartbeat: {stage_name} still running (elapsed: {time.time()-t0:.1f}s)...")
            send_toast(f"⏳ {stage_name} in progress ({int(time.time()-t0)}s)", kind="info")
            
    proc.stdout.close()
    return_code = proc.wait()
    elapsed = time.time() - t0
    
    if return_code != 0:
        log(f"❌ [FAILED STAGE: {stage_name}] (Exit code: {return_code}, Time: {elapsed:.1f}s)")
        send_toast(f"❌ Failed: {stage_name}", kind="danger")
        raise RuntimeError(f"Stage {stage_name} failed with code {return_code}")
        
    log(f"✅ [COMPLETED STAGE: {stage_name}] in {elapsed:.1f}s")
    send_toast(f"✅ Completed: {stage_name} ({elapsed:.1f}s)", kind="success")
    return elapsed

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, "saved_models"), exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, "checkpoints"), exist_ok=True)
    
    with open(LOG_PATH, "w") as f:
        f.write(f"=== CASA Dev Pipeline Started at {time.ctime()} ===\n")
        
    total_start = time.time()
    log("Starting Comprehensive Development Benchmark Pipeline on Blackwell RTX PRO 6000...")
    
    try:
        # Stage 1: Clean Model Training - WideResNet-28-10 (10 epochs)
        log("--> Stage 1/5: Clean Model Training (WideResNet-28-10)...")
        run_command_stream(
            "python3 scripts/train_clean.py --config configs/train/clean_cifar10_wrn28_10.yaml --epochs 10",
            "Clean Training WRN-28-10"
        )
        
        # Stage 2: Adversarial Training - SPGD-AT & PGD-AT (10 epochs each)
        log("--> Stage 2/5: Adversarial Training (SPGD-AT & PGD-AT)...")
        run_command_stream(
            "python3 scripts/train_adversarial.py --config configs/adv_train/spgd_at_cifar10_resnet18.yaml --epochs 10",
            "Adversarial Training SPGD-AT"
        )
        run_command_stream(
            "python3 scripts/train_adversarial.py --config configs/adv_train/pgd_at_cifar10_resnet18.yaml --epochs 10",
            "Adversarial Training PGD-AT"
        )
        
        # Stage 3: Attack Benchmarks (1,000 samples, 11 Attacks)
        log("--> Stage 3/5: Comprehensive Attack Benchmark (1,000 samples)...")
        run_command_stream(
            "python3 scripts/attack_benchmark.py --config configs/development.yaml --strict",
            "Attack Benchmark (Dev 1k)"
        )
        
        # Stage 4: Defense Benchmarks (1,000 samples, 4 Defenses, Oblivious & Adaptive)
        log("--> Stage 4/5: Comprehensive Defense Benchmark (1,000 samples)...")
        run_command_stream(
            "python3 scripts/defense_benchmark.py --config configs/development.yaml --strict",
            "Defense Benchmark (Dev 1k)"
        )
        
        # Stage 5: Ablation Studies (10 Variants, 1,000 samples)
        log("--> Stage 5/5: CASA 10-Variant Component Ablation Study (1,000 samples)...")
        run_command_stream(
            "python3 scripts/run_ablation.py --samples 1000 --k-values 1 4 16 --output result/ablation_results.json --report-md result/ablation_study.md",
            "Ablation Studies (10 Variants)"
        )
        
        total_time = time.time() - total_start
        log(f"🎉🎉 ALL EXPERIMENT STAGES COMPLETED SUCCESSFULLY IN {total_time:.1f}s! 🎉🎉")
        send_toast(f"🎉 All Experiments Completed! Total time: {int(total_time/60)}m", kind="success")
        
        # Mark completion flag
        with open(FLAG_PATH, "w") as f:
            f.write(json.dumps({
                "status": "SUCCESS",
                "total_time_seconds": total_time,
                "timestamp": time.time(),
                "date": time.ctime()
            }))
            
    except Exception as e:
        log(f"FATAL PIPELINE ERROR: {e}\n{traceback.format_exc()}")
        with open(FLAG_PATH, "w") as f:
            f.write(json.dumps({
                "status": "FAILED",
                "error": str(e),
                "timestamp": time.time()
            }))
        sys.exit(1)

if __name__ == "__main__":
    main()
