# Cascading KV Cache

# TODO:

- booksum
  - rerun ours

- PG19
  - token selection ablation (use scores vs no scores) (PG19)
  - re run all of ours
  - run vanilla 16, 32k strided

- Passkey
  - run 32k wiundow
  - run 65K window? (youngwan will run, touch up the files and send him the dataset)

- LongBench
    - move our outdated results to new foler
    - run llama 
    - run qwen

- MMLU
    - run our llama and qwen

- attention matrix plotting.
  - plot new ones with eager fill to see if there is any difference.

- longer context benchmarks
  - LOFT?

- add more baselines
- note in new paper about beta 0.999
  - also note about the approximate score of flash attention?
  - note about LLM lost in the middle paper (cite it and give inspiration in intro)


## How to Install

```bash
conda env create -f environment.yml
pip install -e .

# test that triton can compile cache correctly
python cascade/models/sink_cache_cascade.py
```

## Run Passkey

```
./test-passkey.bash -m llama3.1-8b-instruct -d sink -g [GPU INDEX] -c [CASCADE NUMBER]

./test-passkey.bash -m llama3.1-8b-instruct -d sink -g [GPU INDEX] -c 1
./test-passkey.bash -m llama3.1-8b-instruct -d sink -g [GPU INDEX] -c 8
```

## RUN PG19

```
# set parameters for desired experiment in ./test-pg19.bash
./test-pg19.bash
```

## Run LongBench

```
cd third_party/LongBench-timber/

# edit run.sh to run the correct models/datasets
./run.sh
```

## Run MMLU

```
# set parameters for desired experiment in ./test-mmlu.bash
./test-mmlu.bash
```
