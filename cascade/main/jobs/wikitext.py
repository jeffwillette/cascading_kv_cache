import os
import gc
import torch
from cascade.dataset.pg19 import PG19Streaming
from tqdm import tqdm
import json
from cascade.models.cascading_cache import CascadingKVCache
from transformers.cache_utils import DynamicCache
import math
import numpy as np
from cascade.utils.other import pad_targets
from datasets import load_dataset


def job_ppl_wikitext2(args, model, tokenizer, device):
    stride = args.cascade_stride
    mdl = model.model

    # all_nll = []  # for hyper attention loop
    stats = {"nll-total": 0, "count-total": 0, "ppl-book": {}}

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    string = ""
    for x in data:
        string += x['text']

    input_ids = tokenizer(string, return_tensors="pt", truncation=False).input_ids
    input_ids = input_ids.cuda()
    target_ids = input_ids.clone()

    past_key_values = None
    use_cache = False

    if args.method == "sink":
        use_cache = True

        max_seq_len = mdl.config._window
        max_seq_len = min(max_seq_len, mdl.config.max_position_embeddings // 2)

        print(f"{max_seq_len=}")
        window = max_seq_len // args.cascades

        past_key_values = CascadingKVCache(
            window,
            num_sink_tokens=mdl.config._sinks,
            max_batch_size=mdl.config._batch_size,
            heads=mdl.config.num_key_value_heads // args.world_size,
            dim=mdl.config.hidden_size // mdl.config.num_attention_heads,
            max_seq_len=max_seq_len,
            dtype=torch.float16,
            device=mdl.embed_tokens.weight.device,
            cascade_func=mdl.config._cascade_func,
            head_reduction=mdl.config._head_reduction,
            layers=len(mdl.layers),
        )
    elif args.method == "vanilla":
        max_seq_len = args.window
    elif args.method in ["bigbird", "snapkv", "h2o"]:
        mdl = model.model
        max_seq_len = args.window
        print(f"{max_seq_len=}")
        for lyr in mdl.layers:
            if args.method == "bigbird":
                lyr.self_attn.config._bb_window = args.window
            elif args.method == "snapkv":
                lyr.self_attn.config.max_capacity_prompt = args.window
            elif args.method == "h2o":
                lyr.self_attn.kv_cache.recent_size = args.window // 2
                lyr.self_attn.kv_cache.hh_size = args.window // 2

                lyr.self_attn.kv_cache.cache_size = \
                    lyr.self_attn.kv_cache.recent_size + lyr.self_attn.kv_cache.hh_size

                lyr.self_attn.kv_cache._clean_scores()
                use_cache = True
                past_key_values = DynamicCache()

    else:
        raise ValueError(f"unsupported method: {args.method=}")

    with tqdm(range(0, input_ids.size(1) - 1, stride), ncols=150) as pbar:
        for i in pbar:
            with torch.no_grad():

                inputs = input_ids[:, i:i + stride]
                targets = target_ids[:, i + 1:i + stride + 1]
                targets = pad_targets(inputs, targets, ignore_index=-100)

                # there were some errors on bigbird, so added padding to powers of 2
                # helped. Probably a triton issue
                if args.method == "bigbird":
                    if np.log2(inputs.size(1)) % 1 != 0:
                        target_len = 2 ** math.ceil(np.log2(inputs.size(1)))
                        len_diff = target_len - inputs.size(1)
                        print(f"{inputs.size()=} {target_len=} {len_diff=}")
                        inputs = torch.cat((
                            inputs,
                            torch.zeros(inputs.size(0), len_diff, dtype=inputs.dtype, device=inputs.device)), dim=-1)
                        targets = torch.cat((
                            targets,
                            torch.full((targets.size(0), len_diff), -100, dtype=inputs.dtype, device=inputs.device)), dim=-1)

                output = model(inputs, use_cache=use_cache, past_key_values=past_key_values)
                past_key_values = output.past_key_values
                logits = output.logits

                _nll = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).cpu().float(),
                    targets.reshape(-1).cpu(),
                    ignore_index=-100,
                    reduction="none",
                )

                nll, count = _nll.sum().item(), (targets >= 0).sum().item()
                del logits, output

                stats["nll-total"] += nll
                stats["count-total"] += count

                pbar.set_description(f"ppl: {np.exp(stats['nll-total'] / stats['count-total'])}")

        del past_key_values, targets, inputs, input_ids, target_ids
        # del logits, output
        torch.cuda.empty_cache()
        gc.collect()

    print(f"final ppl: {np.exp(stats['nll-total'] / stats['count-total'])}")
    os.makedirs('./cache/llama_eval/wikitext/', exist_ok=True)
    f = f'./cache/llama_eval/wikitext/{args.method}-{args.model}-{args.comment}-window-{args.window}-' + \
        f'cascades-{args.cascades}-head-reduction-{args.head_reduction}-cascade-func-{args.cascade_func}-' + \
        f'cascade-stride-{args.cascade_stride}-homogeneous-heads-{args.homogeneous_heads}-comment-{args.comment}.json'

    with open(f, 'w') as f:
        json.dump(stats, f)
