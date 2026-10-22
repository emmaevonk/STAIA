
## ⬇️ Installation
Currently, STAIA is not yet available on Pypi. 

Installing from the source:
```bash
conda create -n "STAIA" python=3.11 ipykernel
conda activate STAIA
export PYTHONNOUSERSITE=1

git clone https://github.com/emmaevonk/STAIA.git
cd STAIA
# you can either install the `requirements_CPU` file, or `requirements_GPU` file. 
pip install -r requirements/requirements_CPU.txt --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
pip install "pybanksy[notebooks]" --no-cache-dir
pip install -e . --no-cache-dir
python -m ipykernel install --user --name STAIA --display-name "STAIA"
```
If a GPU is available, change the `requirements/requirements_CPU.txt` to `requirements/requirements_GPU.txt` 

Minimum requirements: look at requirements_v2.txt file 

After running installation, STAIA can be imported by running `import STAIA`.
