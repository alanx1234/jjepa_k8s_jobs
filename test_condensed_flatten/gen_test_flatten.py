#!/usr/bin/env python3
from pathlib import Path

IMAGE = "gitlab-registry.nrp-nautilus.io/jmduarte/hbb_interaction_network:latest"
OPTION_FILE = "/config/ParT_B_amp_1p.json"
TEST_DIR = "/j-jepa-vol/J-JEPA/data/top/test/"
CONFIG_CM = "ptcl-options-amp-1p"

# Parent of the size_key dirs used in training
MODEL_BASE = "/j-jepa-vol/J-JEPA-Alan/model_performances_run2/flatten"

SIZES = ["1k", "10k", "100k", "1m"]


def emit_test_job_yaml(size_key: str) -> str:
    """
    Build a single *test* Job YAML that, for this size_key, loops over:
      - baseline/
      - finetune/* (whatever percentages exist),
    and for each parent-dir, runs test_eval_ptcl with
      --checkpoint-type best_acc and best_rej.
    """
    job_name = f"alan-ptcl-flatten-{size_key}-jets-test-all"
    size_root = f"{MODEL_BASE}/{size_key}"

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

          ROOT="{size_root}"

          # Loop over baseline and any finetune/* dirs that exist
          for parent_dir in "$ROOT/baseline" "$ROOT/finetune"/*; do
            if [ ! -d "$parent_dir" ]; then
              echo "Skipping missing or non-dir: $parent_dir"
              continue
            fi

            echo "=========================================="
            echo "Parent dir: $parent_dir"
            echo "=========================================="

            for ckpt_type in best_acc best_rej; do
              echo "Running test_eval_ptcl on $parent_dir with checkpoint-type=$ckpt_type"
              python -u -m src.evaluation.test_eval_ptcl \\
                --option-file {OPTION_FILE} \\
                --test-dataset-path {TEST_DIR} \\
                --batch-size 256 --sum 0 --flatten 1 --cls 0 \\
                --parent-dir "$parent_dir" \\
                --checkpoint-type "$ckpt_type"
            done
          done
        resources:
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

    for size_key in SIZES:
        fname = out_dir / f"alan-ptcl-flatten-{size_key}-jets-test-all.yaml"
        fname.write_text(emit_test_job_yaml(size_key))
        written.append(str(fname))

    print("Wrote files:")
    for f in written:
        print("  -", f)


if __name__ == "__main__":
    main()  