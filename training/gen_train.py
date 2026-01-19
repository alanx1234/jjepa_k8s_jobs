#!/usr/bin/env python3
import textwrap

def make_job_yaml(
    name_suffix,
    pct_str,
    use_full_train,
    nproc_per_node,
    gpus,
    cpus,
    num_jets,
    mem_gi
):
    """
    name_suffix: '1p', '5p', '10p', '50p', '100p'
    pct_str: '1%', '5%', '10%', '50%', '100%'
    use_full_train: True for 100% (no '100%' in data_path), False otherwise
    """
    job_name = f"alan-part-jjepa-{name_suffix}"
    config_map = f"ptcl-options-amp-{name_suffix}"
    config_json = f"/config/ParT_B_amp_{name_suffix}.json"

    if use_full_train:
        data_path = "/j-jepa-vol/J-JEPA/data/JetClass/ptcl/train"
    else:
        data_path = f"/j-jepa-vol/J-JEPA/data/JetClass/ptcl/{pct_str}/train"

    output_dir = f"/j-jepa-vol/J-JEPA-Alan/models/JetClass/ptcl_filtered/{pct_str}"

    yaml = f"""\
apiVersion: batch/v1
kind: Job
metadata:
  namespace: cms-ml
  labels:
    jobgroup: jjepa-job
  name: {job_name}
spec:
  backoffLimit: 0
  template:
    spec:
      tolerations:
      - key: "nautilus.io/hardware"
        operator: "Equal"
        value: "a100"
        effect: "NoSchedule"
      - key: "node.kubernetes.io/not-ready"
        operator: "Exists"
        effect: "NoExecute"
        tolerationSeconds: 300
      - key: "node.kubernetes.io/unreachable"
        operator: "Exists"
        effect: "NoExecute"
        tolerationSeconds: 300
      initContainers:
      - name: init-clone-repo
        image: alpine/git
        command:
        - /bin/sh
        - -c
        args:
        - |
          git clone --single-branch --branch ptcl_alan https://github.com/alanx1234/J-JEPA.git /opt/repo/J-JEPA && \\
          chown -R 1000:1000 /opt/repo/J-JEPA
        volumeMounts:
        - name: git-repo
          mountPath: /opt/repo
        resources:
          limits:
            cpu: '1'
            memory: 1Gi
          requests:
            cpu: '1'
            memory: 1Gi
      containers:
      - name: testing
        image: nvcr.io/nvidia/pytorch:24.08-py3
        env:
        - name: TORCH_CUDA_ALLOC_CONF
          value: "expandable_segments:True,max_split_size_mb:128"
        - name: NVIDIA_DRIVER_CAPABILITIES
          value: "compute,utility"

        command: ["/bin/bash", "-c"]
        args:
          - |
            cd /opt/repo/J-JEPA
            python -m pip install --upgrade pip wheel setuptools
            pip install --no-cache-dir numpy tqdm h5py matplotlib awkward numba vector fastjet
            pip install -e .

            torchrun --standalone --nnodes=1 --nproc_per_node={nproc_per_node} \\
              src/models/train_model_ptcl.py \\
                --config {config_json} \\
                --num_jets {num_jets} \\
                --data_path {data_path} \\
                --output_dir {output_dir} \\
                --batch_size 256 \\
                --probe \\
                --probe_every 1 \\
                --probe_steps 1000 \\
                --probe_train_jets 50000 \\
                --probe_val_jets 50000 \\
                --probe_lr 1e-2
        resources:
          limits:
            cpu: '{cpus}'
            memory: {mem_gi}Gi
            nvidia.com/a100: {gpus}
          requests:
            cpu: '{cpus}'
            memory: {mem_gi}Gi
            nvidia.com/a100: {gpus}
        volumeMounts:
        - name: git-repo
          mountPath: /opt/repo
        - name: j-jepa-vol
          mountPath: /j-jepa-vol
        - name: config
          mountPath: /config
          readOnly: true
        - name: dshm
          mountPath: /dev/shm
        ports:
        - containerPort: 6006
      volumes:
      - name: git-repo
        emptyDir: {{}}
      - name: j-jepa-vol
        persistentVolumeClaim:
          claimName: j-jepa-vol
      - name: config
        configMap:
          name: {config_map}
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: 64Gi
      restartPolicy: Never
"""
    return textwrap.dedent(yaml)


def main():
    jobs = [
        # name_suffix, pct_str, use_full_train, nproc_per_node, gpus, cpus, num_jets, mem_gi
        # datasets: 1M, 5M, 10M, 50M, 100M; keep 20% → 0.2M, 1M, 2M, 10M, 20M
        ("1p",   "1%",   False, 1, 1, 4,   200_000, 64),
        ("5p",   "5%",   False, 1, 1, 4, 1_000_000, 64),
        ("10p",  "10%",  False, 1, 1, 6, 2_000_000, 128),
        ("50p",  "50%",  False, 2, 2, 8,10_000_000, 256),
        ("100p", "100%", True,  2, 2, 8,20_000_000, 456),
    ]

    for name_suffix, pct_str, use_full_train, nproc, gpus, cpus, num_jets, mem_gi in jobs:
        yaml_text = make_job_yaml(
            name_suffix=name_suffix,
            pct_str=pct_str,
            use_full_train=use_full_train,
            nproc_per_node=nproc,
            gpus=gpus,
            cpus=cpus,
            num_jets=num_jets,
            mem_gi = mem_gi
        )
        filename = f"alan-part-jjepa-{name_suffix}.yaml"
        with open(filename, "w") as f:
            f.write(yaml_text)
        print(f"Wrote {filename}")


if __name__ == "__main__":
    main()
