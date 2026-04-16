#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""兼容入口：已迁移至 src/image_preprocessing/run_preprocess.py。"""

from __future__ import annotations

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.image_preprocessing.run_preprocess import main


if __name__ == "__main__":
    main()
