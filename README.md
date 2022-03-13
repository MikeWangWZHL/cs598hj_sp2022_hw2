# Set up environment
```
conda env create -f environment.yml
```
```
conda activate CS598HJ_HW2
```

# Set up dataset
At the root directory, run

```
git clone https://github.com/di-dimitrov/SEMEVAL-2021-task6-corpus.git
```

# Quick Start
To finetune the pre-trained bert with additional symbolic features, go to `src/`, run:

```
python3 train_bert.py
```

For more detailed usage, e.g., performing hyper-parameter search for the best threshold on dev set, or running random baseline, please refer to the instructions and examples in `__main__` in `train_bert.py`.
