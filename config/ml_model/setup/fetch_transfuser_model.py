"""
Fetches only the tfv4_l6_0 checkpoint -- TransFuser++ v4 trained on the
longest6 benchmark, seed 0 (config.pickle + model_0030.pth) -- from the PCLA
weights dataset on Hugging Face.

These weights were sanity-checked with config/ml_model/setup/test_transfuser.py: loading
this config/checkpoint into LidarCenterNet and running a forward pass with
random dummy input tensors (shaped from the model's own saved config)
completed successfully and produced the expected set of output tensors,
confirming the weight structure matches the model code.

PCLA no longer hosts a single pretrained.zip archive; weights live as
per-agent folders in the dataset repo. Uses `huggingface_hub.snapshot_download`
with `allow_patterns` to pull down only the tfpp_all_0 folder instead of the
whole dataset.

Destination matches what PCLA/agents.json expects for agent string "tfv4_l6_0":
    PCLA/pcla_agents/transfuserv4_pretrained/longest6/tfpp_all_0/
"""

import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "MasoudJTehrani/PCLA"
AGENT_SUBPATH = "transfuserv4_pretrained/longest6/tfpp_all_0"
MEMBERS = [
    f"{AGENT_SUBPATH}/config.pickle",
    f"{AGENT_SUBPATH}/model_0030.pth",
]

def main():
    project_root = Path(__file__).resolve().parents[3]
    pcla_dir = os.path.join(project_root, "PCLA")
    target_dir = os.path.join(pcla_dir, "pcla_agents")

    if all(os.path.exists(os.path.join(target_dir, member)) for member in MEMBERS):
        print("config.pickle and model_0030.pth already exist. skipping...")
        return

    print(f"Downloading {AGENT_SUBPATH} from {REPO_ID}...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=[f"{AGENT_SUBPATH}/*"],
        local_dir=target_dir,
    )

    print("Done.")

if __name__ == "__main__":
    main()
