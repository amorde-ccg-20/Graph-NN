# Setup
```bash
git clone https://github.com/amorde-ccg-20/Graph-NN.git
cd Graph-NN
python -m pip install numpy pandas matplotlib h5py tables tqdm kaggle
kaggle auth login
```

# Download dataset
```bash
kaggle datasets download liuxu77/largest -f ca_his_raw_2020.h5 -p q3/dataset/raw --unzip
kaggle datasets download liuxu77/largest -f ca_his_raw_2021.h5 -p q3/dataset/raw --unzip
kaggle datasets download liuxu77/largest -f ca_meta.csv -p q3/dataset/raw --unzip
kaggle datasets download liuxu77/largest -f ca_rn_adj.npy -p q3/dataset/raw --unzip
```

# Run Q1 and Q2
```bash
python notebooks/q1b.py
python notebooks/q2.py
```

# Prepare Q3 data and graph
```bash
python -m q3.scripts.q3a --metadata q3/dataset/raw/ca_meta.csv --year-2020 q3/dataset/raw/ca_his_raw_2020.h5 --year-2021 q3/dataset/raw/ca_his_raw_2021.h5 --aggregation sum
python -m q3.data_processing.handle_missing
python -m q3.graph.analyze --sensor 1119780 --center exclude
```

# Train Q4 and Q5 on a CUDA GPU
```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
python -u -m q3.scripts.q4
python -u -m q3.scripts.q5
```

# Generate Q6 plots and graph comparison
```bash
python -m q3.scripts.q6
```
