#!/bin/bash
pip install -e ".[cpu]" --no-cache-dir
pip install torch==2.6.0+cpu torch_scatter torch_sparse \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -f https://data.pyg.org/whl/torch-2.6.0+cpu.html --no-cache-dir
pip install "pybanksy[notebooks]" --no-cache-dir