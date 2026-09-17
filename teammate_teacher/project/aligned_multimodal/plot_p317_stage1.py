from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;O=H/"runs/p317_hoi_masked_physical_stage1_v1"
def main():
 d=json.loads((O/"history.json").read_text());plt.figure(figsize=(7,4))
 for n in ("control","hoi"):plt.plot([x["epoch"] for x in d[n]],[x["classification"] for x in d[n]],label=n)
 plt.xlabel("Epoch");plt.ylabel("Classification loss");plt.title("P317 paired Stage-1 classification");plt.legend();plt.tight_layout();plt.savefig(O/"Figure_1.png",dpi=180);plt.close();plt.figure(figsize=(7,4));plt.plot([x["epoch"] for x in d["hoi"]],[x["dynamics"] for x in d["hoi"]],label="dynamics");plt.plot([x["epoch"] for x in d["hoi"]],[x["reconstruction"] for x in d["hoi"]],label="reconstruction");plt.xlabel("Epoch");plt.ylabel("Auxiliary loss");plt.title("P317 HOI auxiliary objectives");plt.legend();plt.tight_layout();plt.savefig(O/"Figure_2.png",dpi=180);plt.close()
if __name__=="__main__":main()
