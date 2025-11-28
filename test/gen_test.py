#!/usr/bin/env python3
from pathlib import Path
from typing import Optional

IMAGE = "gitlab-registry.nrp-nautilus.io/jmduarte/hbb_interaction_network:latest"
OPTION_FILE = "/config/ParT_B_amp_1p.json"
TEST_DIR = "/j-jepa-vol/J-JEPA/data/top/test/"
CONFIG_CM = "ptcl-options-amp-1p"

# Parent of the size_key dirs used in training
MODEL_BASE = "/j-jepa-vol/J-JEPA/model_performances/top/ptcl"

SIZES = {
    "1k": 1_000,
    "10k": 10_000,
    "100k": 100_000,
    "1m": 1_000_000,
}

PRETRAIN_PCTS = ["1", "5", "10", "50", "100"]  # percent values as strings


def emit_test_job_yaml(size_key: str, pct: Optional[str], ckpt_type: Optional[str]) -> str:
    """
    Build a single *test* Job YAML string.
    pct:
      - '1','5','10','50','100' for finetune jobs
      - None for baseline
    ckpt_type:
      - 'best_acc', 'best_rej', or None (for default "last")
    """
    if pct is None:
        parent_subdir = "baseline"
        if ckpt_type == "best_acc":
            tag = "best-acc"
        elif ckpt_type == "best_rej":
            tag = "best-rej"
        else:
            tag = "last"
        job_name = f"alan-ptcl-{size_key}-jets-baseline-{tag}-test"
    else:
        pct_name = f"{pct}p"       # for k8s object name
        pct_path = f"{pct}%"       # for filesystem paths
        parent_subdir = f"finetune/{pct_path}"
        if ckpt_type == "best_acc":
            tag = "best-acc"
        elif ckpt_type == "best_rej":
            tag = "best-rej"
        else:
            tag = "last"
        job_name = f"alan-ptcl-{size_key}-jets-finetune-{pct_name}-{tag}-test"

    parent_dir = f"{MODEL_BASE}/{size_key}/{parent_subdir}"

    # Build the last lines of the python args, with optional --checkpoint-type
    if ckpt_type is None:
        parent_line = f"            --parent-dir {parent_dir}\n"
        ckpt_line = ""
    else:
        parent_line = f"            --parent-dir {parent_dir} \\\n"
        ckpt_line = f"            --checkpoint-type {ckpt_type}\n"

    yaml = f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: cms-ml
  labels: {{ jobgroup: jjepa-job }}
spec:
  completions: 1
  parallelism: 1
  backoffLimit: 5
  template:
    spec:
      restartPolicy: Never
      tolerations:
        - key: "nautilus.io/hardware"
          operator: "Equal"
          value: "gpu"
          effect: "NoSchedule"
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: kubernetes.io/hostname
                operator: NotIn
                values:
                  - ry-gpu-15.sdsc.optiputer.net
                  - gpn-fiona-mizzou-7.rnet.missouri.edu
                  - prp-gpu-3.t2.ucsd.edu
      initContainers:
      - name: init-clone-repo
        image: alpine/git
        command: ["/bin/sh","-c"]
        args:
        - |
          git clone --single-branch --branch ptcl_alan https://github.com/alanx1234/J-JEPA.git /opt/repo/J-JEPA &&
          chown -R 1000:1000 /opt/repo
        resources:
          requests: {{ cpu: "1", memory: 1Gi, ephemeral-storage: "1Gi" }}
          limits:   {{ cpu: "1", memory: 1Gi,  ephemeral-storage: "4Gi" }}
        volumeMounts:
        - {{ name: git-repo, mountPath: /opt/repo }}
      containers:
      - name: runner
        image: {IMAGE}
        env:
        - {{ name: PYTHONPATH, value: "/opt/repo/J-JEPA" }}
        - {{ name: TORCH_CUDA_ALLOC_CONF, value: "max_split_size_mb:128" }}
        - name: POD_NAME
          valueFrom: {{ fieldRef: {{ fieldPath: metadata.name }} }}
        command: ["/bin/bash","-lc"]
        args:
        - |
          set -euo pipefail
          cd /opt/repo/J-JEPA/
          pip install -e .

          python -u -m src.evaluation.test_eval_ptcl \\
            --option-file {OPTION_FILE} \\
            --test-dataset-path {TEST_DIR} \\
            --batch-size 256 --sum 0 --flatten 1 --cls 0 \\
{parent_line}{ckpt_line}        resources:
          requests: {{ cpu: "2", memory: 64Gi, nvidia.com/gpu: 1, ephemeral-storage: "1Gi" }}
          limits:   {{ cpu: "2", memory: 64Gi, nvidia.com/gpu: 1, ephemeral-storage: "16Gi" }}
        volumeMounts:
        - {{ name: git-repo,   mountPath: /opt/repo }}
        - {{ name: j-jepa-vol, mountPath: /j-jepa-vol }}
        - {{ name: config,     mountPath: /config, readOnly: true }}
      volumes:
      - {{ name: git-repo, emptyDir: {{}} }}
      - {{ name: j-jepa-vol, persistentVolumeClaim: {{ claimName: j-jepa-vol }} }}
      - {{ name: config,   configMap: {{ name: {CONFIG_CM} }} }}
"""
    return yaml


def main():
    out_dir = Path(".")
    written = []

    for size_key, _num in SIZES.items():
        # finetune jobs: best_acc and best_rej per pct
        for pct in PRETRAIN_PCTS:
            for ckpt_type, suffix in [("best_acc", "best-acc"), ("best_rej", "best-rej")]:
                fname = out_dir / f"alan-ptcl-{size_key}-jets-finetune-{pct}p-{suffix}-test.yaml"
                fname.write_text(emit_test_job_yaml(size_key, pct, ckpt_type))
                written.append(str(fname))

        # baseline jobs: best_acc and best_rej
        for ckpt_type, suffix in [("best_acc", "best-acc"), ("best_rej", "best-rej")]:
            fname = out_dir / f"alan-ptcl-{size_key}-jets-baseline-{suffix}-test.yaml"
            fname.write_text(emit_test_job_yaml(size_key, None, ckpt_type))
            written.append(str(fname))

    print("Wrote files:")
    for f in written:
        print("  -", f)


if __name__ == "__main__":
    main()
