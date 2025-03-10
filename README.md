# Cascading KV Cache

![Cascading KV Cache](./cascading-cache.jpg)

Cascading KV Cache is a fast and dynamic key-value cache which can be added to pretrained transformers. Our method
converts a pretrained quadratic transformers into one with linear inference complexity. 

| [**Paper (Arxiv, *latest*)**](https://arxiv.org/abs/2406.17808) | [**Paper (ICLR 2025)**](https://openreview.net/forum?id=dSneEp59yX)

## How to Install

```bash
pip install -e .

# run tests
python -m unittest -k [test_name_regex]
```

## Run Passkey

For passkey, batch size must evenly divide 20 (1, 2, 4, 5, 10, 20)

```
./test-passkey.bash -m [MODEL NAME] -d [METHOD] -g [GPU INDEX] -w [WINDOW SIZE] -c [CASCADE NUMBER] -b [BATCH SIZE]
./test-passkey.bash -m llama3.1-8b-instruct -d sink -g [GPU INDEX] -w [WINDOW SIZE] -c 1 -b 2
./test-passkey.bash -m llama3.1-8b-instruct -d sink -g [GPU INDEX] -w [WINDOW SIZE] -c 8 -b 2
```

## RUN PG19

```
# set parameters for desired experiment in ./test-pg19.bash
./test-pg19.bash -m [MODEL NAME] -d [METHOD] -g [GPU INDEX]
./test-pg19.bash -m llama3.1-8b -d sink -g 0
./test-pg19.bash -m qwen2-7b -d sink -g 0
```

## Run LongBench

```
cd third_party/LongBench-timber/

./run.sh - m [MODEL] -d [METHOD] -g [GPU INDEX]

./run.sh -m llama3.1-8b-instruct -d sink -g 0
./run.sh -m llama3.1-8b-instruct -d vanilla -g 0
./run.sh -m qwen2-7b-instruct -d sink -g 0
./run.sh -m qwen2-7b-instruct -d vanilla -g 0

```

## Citation

```tex
@article{willette2024training,
  title={Training-Free Exponential Context Extension via Cascading KV Cache},
  author={Willette, Jeffrey and Lee, Heejun and Lee, Youngwan and Jeon, Myeongjae and Hwang, Sung Ju},
  journal={arXiv preprint arXiv:2406.17808},
  year={2024}
}
```
