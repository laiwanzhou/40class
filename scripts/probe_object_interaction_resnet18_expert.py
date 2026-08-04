from __future__ import annotations

import sys

from src.train_object_interaction_resnet18_expert import main


if __name__ == "__main__":
    if "--probe" not in sys.argv:
        sys.argv.append("--probe")
    main()
