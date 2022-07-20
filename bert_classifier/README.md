# BERT Classifier on the Amazon reviews dataset

* Setup a conda environment and install dependencies

```
conda create env -n torch_dev python=3.9
conda activate torch_dev
pip install torch numpy tqdm transformers datasets sklearn
```

* Train the model: `python train.py`