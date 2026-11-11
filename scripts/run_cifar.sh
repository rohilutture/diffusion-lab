#!/usr/bin/env bash
# Trains the flow and VP CIFAR-10 models back to back (one GPU job at a time), then computes FID vs NFE for both.
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python.exe
$PY -m diffusionlab.train --config configs/cifar_flow.yaml > results/cifar_flow_train.log 2>&1
$PY -m diffusionlab.train --config configs/cifar_vp.yaml > results/cifar_vp_train.log 2>&1
$PY -m diffusionlab.eval_fid --ckpt runs/cifar_flow/ckpt.pt --n 10000 --nfes 10,20,50,100 > results/fid_flow.txt 2>&1
$PY -m diffusionlab.eval_fid --ckpt runs/cifar_vp/ckpt.pt --n 10000 --nfes 10,20,50,100 > results/fid_vp.txt 2>&1
echo done > results/CIFAR_DONE
