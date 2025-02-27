from transformers import LlamaForCausalLM as OriginalLlamaForCausalLM
from cascade.models.cascade_attention import sample_monkeypatch
from tqdm import tqdm
from cascade.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig
import json
from cascade.models.cascading_cache import CascadingKVCache
import os
from pathlib import Path
import time
from typing import List, Tuple, Any

import torch
from torch import Tensor
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast
from minference import MInference, get_support_models
from cascade.models.pyramid_kv_monkeypatch import replace_llama as pyramid_kv_replace_llama

from cascade.main.jobs.infinite_bench_eval_utils import (
    dump_jsonl,
    create_prompt,
    load_data,
    get_answer,
    DATA_NAME_TO_MAX_NEW_TOKENS,
)

from cascade.main.jobs.infinite_bench_args import parse_args


MAX_POSITION_ID = int(os.getenv('SEQ_LEN', '128')) * 1024  # Determined by the model
TRUNCATE_LEN = int(os.getenv('SEQ_LEN', '128')) * 1024
print("truncate len: ", TRUNCATE_LEN)


def truncate_input(input: list, max_length: int, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input, add_special_tokens=False)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}, ", end='')
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    tokens = tok.encode(tok.decode(tokens, skip_special_tokens=False), add_special_tokens=False)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    # print(tokens[:20], tokens[-20:])
    assert len_after <= (len_before + 16)
    assert len_after <= (max_tokens + 16)
    return tok.decode(tokens, skip_special_tokens=False), len_after


def chunk_generate(
    model,
    tok,
    texts: List[str],
    max_tokens: int,
    sliding_window: int = MAX_POSITION_ID,
    chunk_size: int = 2500,
    verbose: bool = False,
) -> List[str]:
    """
    Directly performing inference using HF transformers will result in OOM
    when using one A100 GPU. This is because the attention matrix is too large,
    so we chunk the input up and perform forward pass on each chunk to build
    up the KV cache. Note that each token still has to attend to
    all tokens in the past.
    """
    with torch.no_grad():
        """
        input_ids: (b, n)
        attention_mask: (b, n)
        [
            [0, 0, .., 0, 1, 1, ..., 1]
            ...
        ]
        """
        inputs = tok(texts, return_tensors="pt", padding=False, add_special_tokens=False)
        if model is None:
            inputs = inputs.to('cpu')
        else:
            inputs = inputs.to(model.device).to(torch.float16)  # type: ignore
        input_ids: Tensor = inputs.input_ids  # (b, n)

        past_key_values = None
        mdl = model.model
        if os.getenv('METHOD', 'sink') in ["sink", "minference-cascade"]:
            past_key_values = CascadingKVCache(
                mdl.config._window // mdl.config._cascades,
                num_sink_tokens=mdl.config._sinks,
                max_batch_size=mdl.config._batch_size,
                heads=mdl.config.num_key_value_heads // mdl.config.world_size,
                dim=mdl.config.hidden_size // mdl.config.num_attention_heads,
                max_seq_len=mdl.config._window,
                dtype=torch.float16,
                device=mdl.embed_tokens.weight.device,
                cascade_func=mdl.config._cascade_func,
                head_reduction=mdl.config._head_reduction,
                layers=len(mdl.layers),
            )

        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            # eos_token_id=tok.pad_token_id,
            past_key_values=past_key_values,
            use_cache=True,
            do_sample=False,
        )

        responses = [
            tok.decode(t[input_ids.shape[-1]:], skip_special_tokens=True) for t in outputs
        ]
    return responses


def get_pred(
    model,
    tok: AutoTokenizer,
    input_text: str,
    max_tokens: int,
    verbose: bool = False,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    print("Truncating... ", end='')
    # pre_len = len(input_text)
    if os.getenv('METHOD', 'sink') in ["minference", "vanilla", "pyramid_kv"]:
        input_text, len_after = truncate_by_tokens(input_text, tok, 32768)
    else:
        input_text, len_after = truncate_by_tokens(input_text, tok, 10000000)

    # input_text, len_after = truncate_by_tokens(input_text, tok, TRUNCATE_LEN - max_tokens - 32)
    # print(f' {pre_len} -> {len(input_text)}')
    if verbose:
        print("# chars:", len(input_text))
        print("=============== Input ===============")
        print(input_text[:500])
        print("...")
        print(input_text[-500:])
        print("=====================================")
    output = chunk_generate(
        model,
        tok,
        [input_text],
        max_tokens=max_tokens,
        chunk_size=4096,
        verbose=verbose,
    )[0]
    output = output.replace('<|eot_id|>', '')
    output = output.replace('<eos>', '')
    output = output.replace('<end_of_turn>', '')
    output = output.replace('[|endofturn|]', '')
    print("Chunked generation:", output.replace('\n', '\\n'))
    return output, len_after


ATTENTION_METHOD = os.getenv('ATTENTION_METHOD', 'hip')


def load_model(model_name: str) -> Tuple["LlamaForCausalLM", AutoTokenizer]:
    print("Loading tokenizer")
    tok = AutoTokenizer.from_pretrained(model_name)

    if os.getenv('METHOD', 'sink') == "minference":
        config = LlamaConfig.from_pretrained(model_name)
        config.attn_implementation = config._attn_implementation = 'flash_attention_2'
        config._method = "minference"

        model = OriginalLlamaForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map="cpu",
        )

        minference_name = model_name
        if "llama" in minference_name.lower():
            minference_name = "meta-llama/" + model_name.split("/")[-1]
        elif "qwen" in minference_name.lower():
            minference_name = "Qwen/" + model_name.split("/")[-1]
        else:
            raise NotImplementedError("model not implemented for minference")

        model.minference_name = minference_name

        # Patch MInference Module,
        # If you use the local path, please use the model_name from HF when initializing MInference.
        minference_patch = MInference(
            attn_type="minference",
            model_name=minference_name,
        )

        model = minference_patch(model)
        model = model.cuda()
    elif os.getenv('METHOD', 'sink') == "minference-cascade":
        config = LlamaConfig.from_pretrained(model_name)
        config.attn_implementation = config._attn_implementation = 'flash_attention_2'
        config._method = "minference-cascade"
        config._batch_size = 1
        config._sinks = 64
        config._cascades = int(os.getenv('CASCADES', 1))
        config._window = 32768
        config.world_size = 1
        config._cascade_func = "pow2"
        config._head_reduction = "max"
        config._method = os.getenv("METHOD", "sink")
        config._cascade_stride = 65536 + 32768
        config._homogeneous_heads = False
        config._do_og_pos = False

        model = OriginalLlamaForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map="cpu",
        )

        minference_name = model_name
        if "llama" in minference_name.lower():
            minference_name = "meta-llama/" + model_name.split("/")[-1]
        elif "qwen" in minference_name.lower():
            minference_name = "Qwen/" + model_name.split("/")[-1]
        else:
            raise NotImplementedError("model not implemented for minference")

        model.minference_name = minference_name

        # Patch MInference Module,
        # If you use the local path, please use the model_name from HF when initializing MInference.
        minference_patch = MInference(
            attn_type="minference",
            model_name=minference_name,
            use_cascade=True,
        )

        model = minference_patch(model)
        model = sample_monkeypatch(model)
        model = model.cuda()

    elif os.getenv('METHOD', 'sink') == "vanilla":
        config = LlamaConfig.from_pretrained(model_name)
        config.attn_implementation = config._attn_implementation = 'flash_attention_2'
        config._method = "vanilla"

        model = OriginalLlamaForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map="cpu",
        )

        model = model.cuda()
    elif os.getenv('METHOD', 'sink') == "pyramid_kv":
        print("replacing pyramid kv")
        pyramid_kv_replace_llama("pyramidkv")

        config = LlamaConfig.from_pretrained(model_name)
        config.attn_implementation = config._attn_implementation = 'flash_attention_2'

        model = OriginalLlamaForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.float16,
            device_map="cpu",
        )
        model = model.cuda()
    else:
        # tok.pad_token = tok.eos_token

        config = LlamaConfig.from_pretrained(model_name)

        config._attn_implementation = config.attn_implementation = 'eager'
        if method in ["vanilla", "snapkv", "bigbird"]:
            config._attn_implementation = config.attn_implementation = 'flash_attention_2'

        config._batch_size = 1
        config._sinks = 64
        config._cascades = int(os.getenv('CASCADES', 4))
        config._window = 32768
        config.world_size = 1
        config._cascade_func = "pow2"
        config._head_reduction = "max"
        config._method = os.getenv("METHOD", "sink")
        config._cascade_stride = 4096
        config._homogeneous_heads = False
        config._do_og_pos = False

        print("Loading model")
        start_time = time.time()
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            config=config,
            device_map="cpu",
            torch_dtype=torch.float16,
        )
        model = sample_monkeypatch(model)
        model = model.cuda()

        for m in model.modules():
            if hasattr(m, 'attention_method'):
                m.attention_method = ATTENTION_METHOD
        print("Time taken:", round(time.time() - start_time))

    return model, tok  # type: ignore


if __name__ == "__main__":
    args = parse_args()
    model_name = f"llama3-{TRUNCATE_LEN // 1024}-{args.model_name}"

    print(json.dumps(vars(args), indent=4))
    data_name = args.task

    # Model
    method = os.getenv('METHOD', "sink")
    cascades = os.getenv('CASCADES', 4)
    max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
    model, tok = load_model(args.model_path)

    # Data
    result_dir = Path(args.output_dir, model_name)
    result_dir.mkdir(exist_ok=True, parents=True)
    examples = load_data(data_name, data_dir=args.data_dir)

    comment = "-rebuttal-stride-32p65-window-32768-tenth-topk"
    if args.stop_idx is None:
        args.stop_idx = len(examples)
        output_path = (
            result_dir / f"preds_{data_name}-cascades-{cascades}-{method}{comment}.jsonl"
        )
    else:
        output_path = (
            result_dir / f"preds_{data_name}_{args.start_idx}-{args.stop_idx}-cascades-{cascades}-{method}{comment}.jsonl"  # noqa
        )

    preds = []
    print("==== Evaluation ====")
    print(f"# examples: {len(examples)}")
    print(f"Start index: {args.start_idx}")
    print(f"Stop index: {args.stop_idx}")
    print(f"Verbose: {args.verbose}")
    print(f"Max tokens: {max_tokens}")
    for i in tqdm(range(args.start_idx, args.stop_idx)):
        eg = examples[i]
        input_text = create_prompt(eg, data_name, model_name, args.data_dir)
        print(f"====== Example {i} ======")
        pred, len_after = get_pred(model, tok, input_text, max_tokens=max_tokens, verbose=args.verbose)
        if args.verbose:
            print(pred)

        preds.append(
            {
                "id": i,
                "prediction": pred,
                "ground_truth": get_answer(eg, data_name),
                "context_length": len_after,
            }
        )
        dump_jsonl(preds, output_path)
