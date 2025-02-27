import torch
import json
from cascade.models.cascading_cache import CascadingKVCache
from transformers.cache_utils import DynamicCache
import time
# import deepspeed
# from cascade.main.jobs.pg19 import get_injection_policy


def job_latency(args, model, tokenizer, device):
    m = model.model

    latency = []
    stride = args.cascade_stride
    rng = [2 ** i for i in range(12, 21)]
    if args.method in ["sink", "minference-cascade"]:
        if args.cascade_stride == 1:
            rng = [2 ** i for i in range(12, 16)]
        else:
            rng = [2 ** i for i in range(12, 21)]
    elif args.method == "h2o":
        # truncate because it will go OOM
        rng = [2 ** i for i in range(10, 14)]
    elif args.method == "minference":
        # truncate because it will go OOM
        rng = [2 ** i for i in range(10, 20)]

    print(f"{args.method=}")
    for l in rng:
        input_ids = torch.randn(1, 1024, 4096, dtype=torch.float16, device="cuda").repeat(1, l // 1024, 1)
        position_ids = torch.arange(l, device="cuda").unsqueeze(0)
        position_embeddings = m.rotary_emb(input_ids, torch.arange(l, device="cuda").view(1, -1))
        model = m.layers[0].self_attn

        if args.method in ["sink", "minference-cascade"]:
            window = args.window // args.cascades
            past_key_values = CascadingKVCache(
                window, num_sink_tokens=4, max_batch_size=1,
                heads=m.config.num_key_value_heads, dim=128,
                max_seq_len=args.window,
                dtype=torch.float16,
                device="cuda",
                cascade_func="pow2",
                head_reduction=args.head_reduction,
                layers=1,
            )
        elif args.method in ["h2o", "snapkv"]:
            for lyr in m.layers:
                if args.method == "snapkv":
                    lyr.self_attn.config.max_capacity_prompt = args.window
                elif args.method == "h2o":
                    lyr.self_attn.kv_cache.recent_size = args.window // 2
                    lyr.self_attn.kv_cache.hh_size = args.window // 2

                    lyr.self_attn.kv_cache.cache_size = \
                        lyr.self_attn.kv_cache.recent_size + lyr.self_attn.kv_cache.hh_size

                    lyr.self_attn.kv_cache._clean_scores()

        with torch.no_grad():
            if args.method == "sink":
                for i in range(0, input_ids.size(1), stride):
                    inp = input_ids[:, i:i + stride]
                    output = model(inp, use_cache=True, past_key_value=past_key_values)
                    past_key_values = output[2]

                del output
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                past_key_values.reset()
                tic = time.perf_counter()
                for i in range(0, input_ids.size(1), stride):
                    inp = input_ids[:, i:i + stride]
                    output = model(inp, use_cache=True, past_key_value=past_key_values)
                    past_key_values = output[2]

                torch.cuda.synchronize()
                t = time.perf_counter() - tic
                latency.append(t)
                print(f"latency for sink: len: {l}: {t}")

                del output, input_ids, position_embeddings
                torch.cuda.empty_cache()
            elif args.method == "minference":
                past_key_values = DynamicCache()
                output = model(
                    input_ids,
                    attention_mask=None,
                    position_ids=position_ids,
                    output_attentions=False,
                    past_key_value=past_key_values,
                    use_cache=True
                )
                past_key_values = output[2]

                del output
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                model.last_it = False
                past_key_values = DynamicCache()

                tic = time.perf_counter()
                output = model(
                    input_ids,
                    attention_mask=None,
                    position_ids=position_ids,
                    output_attentions=False,
                    past_key_value=past_key_values,
                    use_cache=True,
                )
                past_key_values = output[2]

                torch.cuda.synchronize()
                t = time.perf_counter() - tic
                latency.append(t)
                print(f"latency for minference: len: {l}: {t}")

                del output, input_ids, position_embeddings
                torch.cuda.empty_cache()

            elif args.method == "minference-cascade":
                model.last_it = False

                for i in range(0, input_ids.size(1), stride):
                    if i + stride >= input_ids.size(1):
                        model.last_it = True

                    inp = input_ids[:, i:i + stride]
                    output = model(
                        inp,
                        attention_mask=None,
                        position_ids=None,
                        output_attentions=False,
                        past_key_value=past_key_values, use_cache=True
                    )
                    past_key_values = output[2]

                del output
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                past_key_values.reset()
                model.last_it = False

                tic = time.perf_counter()
                for i in range(0, input_ids.size(1), stride):
                    if i + stride >= input_ids.size(1):
                        model.last_it = True
                    inp = input_ids[:, i:i + stride]
                    output = model(
                        inp,
                        attention_mask=None,
                        position_ids=None,
                        output_attentions=False,
                        past_key_value=past_key_values, use_cache=True
                    )
                    past_key_values = output[2]

                torch.cuda.synchronize()
                t = time.perf_counter() - tic
                latency.append(t)
                print(f"latency for minference-cascade: len: {l}: {t}")

                del output, input_ids, position_embeddings
                torch.cuda.empty_cache()

            elif args.method == "h2o":
                # warmup to compile and autotune triton if necessary
                past_key_values = DynamicCache()
                output = model(input_ids, position_ids=position_ids, use_cache=True, past_key_value=past_key_values)
                past_key_values = output[2]

                past_key_values = DynamicCache()
                model.kv_cache._clean_scores()
                torch.cuda.synchronize()

                tic = time.perf_counter()
                output = model(input_ids, position_ids=position_ids, use_cache=True, past_key_value=past_key_values)
                past_key_values = output[2]

                torch.cuda.synchronize()
                t = time.perf_counter() - tic
                latency.append(t)
                print(f"latency for h2o: len: {l}: {t}")

            elif args.method == "snapkv":
                cache = DynamicCache()
                output = model(
                    input_ids,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_value=cache,
                    use_cache=True,
                )

                cache = DynamicCache()
                del output, cache
                torch.cuda.synchronize()

                cache = DynamicCache()

                tic = time.perf_counter()
                output = model(
                    input_ids,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_value=cache,
                    use_cache=True,
                )
                torch.cuda.synchronize()
                t = time.perf_counter() - tic
                latency.append(t)
                print(f"latency for snapkv: len: {l}: {t}")

                del output, position_embeddings, input_ids, cache
            else:
                cache = DynamicCache()
                output = model(
                    input_ids,
                    position_embeddings=position_embeddings,
                    past_key_value=cache,
                    use_cache=True,
                )

                cache = DynamicCache()
                del output, cache
                torch.cuda.synchronize()

                cache = DynamicCache()

                tic = time.perf_counter()
                output = model(
                    input_ids,
                    position_embeddings=position_embeddings,
                    past_key_value=cache,
                    use_cache=True,
                )
                torch.cuda.synchronize()
                t = time.perf_counter() - tic
                latency.append(t)
                print(f"latency for vanilla: len: {l}: {t}")

                del output, position_embeddings, input_ids, cache

        torch.cuda.empty_cache()

    with open(f"./plots/latency/attention-{args.method}-stride-{args.cascade_stride}-comment-{args.comment}.json", "w") as f:
        json.dump(latency, f)
