import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from peft import PeftModel

from network import BEVNetwork, ChoiceHead, load_backbone


class RunLogger:
    """Each training run gets runs/<name>-<timestamp>/ with config, metrics and checkpoints (LoRA + head only)."""

    def __init__(self, config, run_name, device, root=None, run_dir=None):
        self.config = config
        if run_dir is not None:
            # --resume-state continues the SAME run dir and appends to its metrics.jsonl
            self.run_dir = Path(run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.exp_id = self.run_dir.name
            self.appended = (self.run_dir / "metrics.jsonl").exists()
        else:
            # RUNS_ROOT lets SageMaker place runs under /opt/ml/output/data, the only
            # directory it uploads. Defaults to ./runs, so local runs are unchanged.
            root = root or os.environ.get("RUNS_ROOT", "runs")
            self.exp_id = f"{run_name}-{datetime.now():%Y%m%d-%H%M%S}"
            self.run_dir = Path(root) / self.exp_id
            self.run_dir.mkdir(parents=True)
            self.appended = False
        self.start_time = time.time()

        config_path = self.run_dir / "config.yaml"
        if not config_path.exists():
            self.write_config(self.run_dir)
        with open(self.run_dir / "run.json", "w") as f:
            json.dump({"exp_id": self.exp_id, "device": str(device),
                       "started_at": datetime.now().isoformat(),
                       "resumed_into": self.appended}, f, indent=2)
        print(f"Experiment {self.exp_id} -> {self.run_dir}")

    def write_config(self, directory):
        with open(directory / "config.yaml", "w") as f:
            yaml.safe_dump(self.config, f, sort_keys=False)

    def log(self, step, **metrics):
        record = {"step": step, "time": round(time.time() - self.start_time, 1), **metrics}
        with open(self.run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(" | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))

    def save_checkpoint(self, network, name="final", step=None):
        """A checkpoint folder is self-contained: copy it anywhere and load_checkpoint() rebuilds the network."""
        ckpt_dir = self.run_dir / "checkpoints" / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(network.backbone, PeftModel):
            network.backbone.save_pretrained(ckpt_dir / "lora")  # adapter weights only, not Qwen itself
        torch.save({k: v.cpu() for k, v in network.head.state_dict().items()}, ckpt_dir / "head.pt")

        self.write_config(ckpt_dir)
        with open(ckpt_dir / "meta.json", "w") as f:
            json.dump({
                "exp_id": self.exp_id,
                "step": step,
                "model_name": self.config["model"]["name"],
                "num_layers": network.backbone.config.num_hidden_layers,
                "head_config": {**self.config["head"], "num_task_types": network.head.task_embedding.num_embeddings},
                "max_state_tokens": self.config["data"]["max_state_tokens"],
                "max_choice_tokens": self.config["data"]["max_choice_tokens"],
                "data_configs": self.config["data"].get("configs", ["default"]),
            }, f, indent=2)
        print(f"Saved checkpoint -> {ckpt_dir}")
        return ckpt_dir

    def save_resume_state(self, network, train_state):
        """Crash-safe rolling save into run_dir/checkpoints/resume/{slot_a,slot_b} + a pointer file.

        Both slots are kept, so the last two saves always exist. Writing goes: clear the UNUSED
        slot, write LoRA + head + train_state.pt into it completely, then flip a single-line
        pointer file (pointer.tmp written first, then os.replace -> atomic on Windows). A crash
        mid-write can only ever damage a slot that is not currently visible, and the previous
        good save stays readable. slots are small (LoRA ranks + 512-input head + optimizer state).
        """
        resume_dir = self.run_dir / "checkpoints" / "resume"
        resume_dir.mkdir(parents=True, exist_ok=True)
        slots = [resume_dir / "slot_a", resume_dir / "slot_b"]
        pointer = resume_dir / "pointer"

        current = pointer.read_text().strip() if pointer.exists() else None
        target = slots[1] if current == "slot_a" else slots[0]  # default to slot_a on first save

        # fresh tmp write first (a partially written slot is never the pointed-to one)
        tmp = resume_dir / (f".tmp_{target.name}")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        if isinstance(network.backbone, PeftModel):
            network.backbone.save_pretrained(tmp / "lora")
        torch.save({k: v.cpu() for k, v in network.head.state_dict().items()}, tmp / "head.pt")
        torch.save(train_state, tmp / "train_state.pt")
        with open(tmp / "meta.json", "w") as f:
            json.dump({"exp_id": self.exp_id, "step": train_state["step"],
                       "saved_at": datetime.now().isoformat()}, f, indent=2)

        # flip the pointer atomically: readers may see the old slot but never a half-written one
        tmp_pointer = resume_dir / "pointer.tmp"
        tmp_pointer.write_text(target.name)
        if tmp.exists():  # move the finished tmp dir into its slot name
            if target.exists():
                shutil.rmtree(target)
            os.replace(tmp, target)
        os.replace(tmp_pointer, pointer)
        return resume_dir

    def latest_resume_state(self):
        """Path to the newest readable resume slot (or None before the first save ever finished)."""
        resume_dir = self.run_dir / "checkpoints" / "resume"
        pointer = resume_dir / "pointer"
        if not pointer.exists():
            return None
        slot = resume_dir / pointer.read_text().strip()
        return slot if (slot / "train_state.pt").exists() else None

    def finish(self, **summary):
        summary["duration_sec"] = round(time.time() - self.start_time, 1)
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def load_checkpoint(ckpt_dir, device):
    ckpt_dir = Path(ckpt_dir)
    with open(ckpt_dir / "meta.json") as f:
        meta = json.load(f)
    # Per-task-type temperatures written next to the checkpoint by calibrate.py, applied in answer()
    calibration = ckpt_dir / "calibration.json"
    if calibration.exists():
        meta["temperatures"] = json.loads(calibration.read_text())["temperatures"]

    backbone = load_backbone(meta["model_name"], meta.get("num_layers"))
    if (ckpt_dir / "lora").exists():
        backbone = PeftModel.from_pretrained(backbone, ckpt_dir / "lora")
    head = ChoiceHead(hidden_dim=backbone.config.hidden_size, **meta["head_config"])
    head.load_state_dict(torch.load(ckpt_dir / "head.pt", map_location="cpu"))
    return BEVNetwork(backbone, head).to(device).eval(), meta
