#!/bin/bash
pip install -e ".[gpu]" --no-cache-dir
pip install torch==2.6.0 torch_scatter torch_sparse \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
pip install "pybanksy[notebooks]"