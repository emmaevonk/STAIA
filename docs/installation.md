
## ⬇️ Installation

### Simplified installation
The installation of STAIA is simplified to running these 3 steps:
```bash
git clone https://github.com/emmaevonk/STAIA.git
cd STAIA
bash scripts/install_cpu.sh   # or install_gpu.sh
```


### Installation CPU
STAIA is meant to be run from a conda environment, so create one:
```bash
conda create -n "STAIA-env" python=3.11 ipykernel
```
The conda environment is built with python 3.11, because the latest version of BANKSY (which is integrated in this package), does not support newer Python versions (yet).
Now it's time to activate the environment and double check the python version present in the environment:
```bash
conda activate STAIA-env
python --version
```
When working on an HPC, the home folder has limited space and can therefore be occupied very fast. Run this command to prevent this from happening when using an HPC (if not, this can be ignored):
```bash
export PYTHONNOUSERSITE=1
```
The STAIA package is not (yet) available on the Python Packaging Index (Pypi), so the repository needs to be cloned from github:
```bash
git clone https://github.com/emmaevonk/STAIA.git
cd STAIA
pip install -r requirements_CPU.txt --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
pip install "pybanksy[notebooks]"
```

Now all necessary packages to run STAIA have been installed. Run `ipykernel` to make sure you can use the environment as a kernel in a Jupyter Notebook:
```bash
python -m ipykernel install --user --name STAIA-env --display-name "STAIA-env"
```

### Installation GPU
STAIA is meant to be run from a conda environment, so create one:
```bash
conda create -n "STAIA-env" python=3.11 ipykernel
```
The conda environment is built with python 3.11, because the latest version of BANKSY (which is integrated in this package), does not support newer Python versions (yet).
Now it's time to activate the environment and double check the python version present in the environment:
```bash
conda activate STAIA-env
python --version
```
When working on an HPC, the home folder has limited space and can therefore be occupied very fast. Run this command to prevent this from happening when using an HPC (if not, this can be ignored):
```bash
export PYTHONNOUSERSITE=1
```
The STAIA package is not (yet) available on the Python Packaging Index (Pypi), so the repository needs to be cloned from github:
```bash
git clone https://github.com/emmaevonk/STAIA.git
cd STAIA
pip install -r requirements_GPU.txt --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
pip install "pybanksy[notebooks]"
```

Now all necessary packages to run STAIA have been installed. Run `ipykernel` to make sure you can use the environment as a kernel in a Jupyter Notebook:
```bash
python -m ipykernel install --user --name STAIA-env --display-name "STAIA-env"
```