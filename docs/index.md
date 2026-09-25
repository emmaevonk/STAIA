# 📦 STAIA
> *STAIA is a Python package for spatial transcriptomics with its aim to lower the computational barrier in spatial transcriptomics analysis. The software is currently focused on Xenium data from 10X Genomics.*

## 🌟 Highlights

- End-to-end analyses in one pipeline
- Recognizing samples from the same Xenium slide
- Assigning labels to samples
- Performing QC, clustering, multimodal segmentation, assigning conditions and pseudobulk DE analysis
- Errorhandler
- Using well known tools in spatial transcriptomics, e.g. Squidpy and Scanpy

## ℹ️ Overview

STAIA is a functional API for analyzing spatial transcriptomics data. It provides a Command Line Interface (CLI) to perform quality control, spatial statistics, neighborhood analysis, and visualization for datasets generated with the Xenium platform from 10X Genomics.

The API is designed around a clean, functional workflow:
```py
STAIA.function()
```

This makes it easy to build reproducible spatial analysis pipelines with minimal boilerplate.

## ✨ Features
STAIA supports a wide range of spatial transcriptomics analyses. Beneat a few examples are mentioned.

### Input/Output
The input and output for every function is mentioned in the documentation. The input is most commonly a SpatialData or an AnnData object.  

### Quality control
- Automatic QC metric calculation
- Data-driven filtering recommendations
- Cell- and gene-level diagnostics

### Spatial Analysis
- Neighbor enrichment / cell-cell interaction analysis
- Spatial neighborhood constructuren
- Spatial statistics and summary metrics

### Visualization
- High-resolution spatial plotting
- Overlay gene expression or cell annotations
- Quick visualization with:
```py
STAIA.plot_image()
```

### Metrics & Diagnostics
- Dataset-level summary statistics
- Spatial structure evaluation
- Analysis-ready feature generation


## License
MIT license is applied:
Copyright (c) 2026 SpatialAPI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


