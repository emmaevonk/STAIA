# 📦 Spatial Transcriptomics AI Agent (STAIA)

> *STAIA is a functional API for spatial transcriptomics. Use it wisely.*


[Documentation](https://emmaevonk.github.io/STAIA/) can be found [here](https://emmaevonk.github.io/STAIA/).

Please note that STAIA is built for python 3.10, 3.11, and 3.12.


## ⬇️ Installation

Currently, STAIA is not yet available on Pypi. 

Installing from the source:
```bash
conda create -n "STAIA" python=3.11 ipykernel
conda activate STAIA
export PYTHONNOUSERSITE=1

git clone https://github.com/emmaevonk/STAIA.git
cd STAIA
pip install -r requirements/requirements_CPU.txt --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
pip install "pybanksy[notebooks]" --no-cache-dir
pip install -e . --no-cache-dir
python -m ipykernel install --user --name STAIA --display-name "STAIA"
```

Minimum requirements: look at requirements_v2.txt file 

After running installation, STAIA can be imported by running `import STAIA`.

## ℹ️ Overview

STAIA is a functional pipeline for analyzing spatial transcriptomics data. It provides a Command Line Interface (CLI) to perform quality control, spatial statistics, neighborhood analysis, and visualization for datasets generated with the Xenium platform from 10X Genomics.

The API is designed around a clean, functional workflow:
```py
STAIA.method()
```

This makes it easy to build reproducible spatial analysis pipelines with minimal boilerplate.


## ✨ Features
STAIA supports a wide range of spatial transcriptomics analyses.

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


## 🚀 Quick Start 
A small example of steps that can be taken with STAIA. 

1. Load Xenium data
```py
import STAIA
sdata = STAIA.read_data(path_xenium)
adata = sdata.tables["table"].copy()
``` 
2. Obtain metrics (percentage transcripts and staining positive pixels allocated)
```py
perc_transcripts, s_pos = STAIA.metrics(sdata)
``` 
3. QC
```py
STAIA.plot_qc_metrics(adata, save=True)
thr = STAIA.recommend_threshold(adata)
adata_filtered = STAIA.perform_quality_control(adata, thr)
```
4. Log normalize adata
```py
sc.pp.normalize_total(adata_filtered, target_sum=1e4)
sc.pp.log1p(adata_filtered)
```
5. Assigning conditions
```py
adata = STAIA.run_watershed(adata, sdata)
df_split_base = df.copy()
session = STAIA.SplitSession(df_split_base)
session.show()
```
6. Perform annotation
7. Analysis
```py
STAIA.dotplot(adata)
STAIA.nhood_enrichment(adata) 
STAIA.celltype_composition(adata)
STAIA.pseudobulk(adata, celltype="Endothelial cells")
```

## 📄 License
MIT license is applied.