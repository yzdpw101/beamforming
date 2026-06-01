"""复合零陷优化 — 同时激活 sidelobe + null_steering 组件。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from examples.beamforming._template.run import main
if __name__ == "__main__":
    main()

