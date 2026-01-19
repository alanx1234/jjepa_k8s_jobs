#!/usr/bin/env python3
from pathlib import Path
from typing import Optional

IMAGE = "gitlab-registry.nrp-nautilus.io/jmduarte/hbb_interaction_network:latest"
OPTION_FILE = "/config/ParT_B_amp_1p.json"
TRAIN_DIR = "/j-jepa-vol/J-JEPA/data/top/train/"
VAL_DIR   = "/j-jepa-vol/J-JEPA/data/top/val/"
CONFIG_CM = "ptcl-options-amp-1p"

SIZES = {
    "1k": 1_000,
    "10k": 10_000,
    "100k": 100_000,
    "1m": 1_000_000,
}

PRETRAIN_PCTS = ["1", "5", "10", "50", "100"]  # percent values as strings


def emit_job_yaml(size_key: str, num_samples: int, pct: Optional[str]) -> str:
    """
    Build a single Job YAML string.
    pct:
      - '1','5','10','50','100' for finetune jobs
      - None for baseline
    """
    if pct is None:
        job_name = f"alan-ptcl-{size_key}-jets-cls-baseline"
        out_subdir = "baseline"
        load_line = ""  # no checkpoint for baseline
        # baseline: ensure previous line has a backslash, then put label on next line
        from_checkpoint_line = "            --from-checkpoint 0 \\\n"
        label_line = "            --label from_scratch\n"
    else:
        pct_name = f"{pct}p"       # for k8s object name
        pct_path = f"{pct}%"       # for filesystem paths
        job_name = f"alan-ptcl-{size_key}-jets-finetune-cls-{pct_name}"
        out_subdir = f"finetune/{pct_path}"
        ckpt_dir = "/j-jepa-vol/J-JEPA-Alan/models/JetClass/ptcl_filtered"
        ckpt_sub = "100%" if pct == "100" else pct_path
        ckpt_path = f"{ckpt_dir}/{ckpt_sub}/best_model.pth"
        load_line = f"            --load-jjepa-path {ckpt_path} \\\n"
        # finetune: no label, and from-checkpoint is terminal (no backslash)
        from_checkpoint_line = "            --from-checkpoint 0\n"
        label_line = ""

    out_dir = f"/j-jepa-vol/J-JEPA-Alan/model_performances_run2/cls/{size_key}/{out_subdir}"

    yaml = f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
  namespace: cms-ml
  labels: {{ jobgroup: jjepa-job }}
spec:
  completions: 5
  parallelism: 5
  completionMode: Indexed
  backoffLimit: 5
  backoffLimitPerIndex: 3 
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
          cd /opt/repo/J-JEPA
          pip install -e .

          python -u -m src.evaluation.finetune_ptcl \\
            --option-file {OPTION_FILE} \\
            --train-dataset-path {TRAIN_DIR} \\
            --val-dataset-path   {VAL_DIR} \\
            --out-dir {out_dir} \\
{load_line}            --batch-size 128 --sum 0 --flatten 0 --cls 1 --finetune 1 \\
            --n-epoch 300 --num-samples {num_samples} \\
{from_checkpoint_line}{label_line}        resources:
          requests: {{ cpu: "4", memory: 64Gi, nvidia.com/gpu: 1, ephemeral-storage: "1Gi" }}
          limits:   {{ cpu: "4", memory: 64Gi, nvidia.com/gpu: 1, ephemeral-storage: "16Gi" }}
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

    for size_key, num in SIZES.items():
        # five finetune jobs
        for pct in PRETRAIN_PCTS:
            fname = out_dir / f"alan-ptcl-{size_key}-jets-finetune-{pct}p.yaml"
            fname.write_text(emit_job_yaml(size_key, num, pct))
            written.append(str(fname))
        # baseline
        fname = out_dir / f"alan-ptcl-{size_key}-jets-baseline.yaml"
        fname.write_text(emit_job_yaml(size_key, num, None))
        written.append(str(fname))

    print("Wrote files:")
    for f in written:
        print("  -", f)


if __name__ == "__main__":
    main()
