import os
import torch
import transformers

from peft import LoraConfig, TaskType
from minference import MInference, get_support_models
from peft import get_peft_model, prepare_model_for_kbit_training
from cascade.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig, LlamaDecoderLayer
from cascade.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Config, Qwen2DecoderLayer
from cascade.models.cascade_attention import sample_monkeypatch
from cascade.models.snapkv import replace_llama
from cascade.utils import seed

from cascade.main.jobs.passkey import job_passkey
from cascade.main.jobs.attn_matrix_plot import job_attn_matrix
from cascade.main.jobs.latency import job_latency
from cascade.main.jobs.ppl import job_ppl
from cascade.main.jobs.pg19 import job_ppl_pg19
from cascade.main.jobs.wikitext import job_ppl_wikitext2
from cascade.main.jobs.booksum import job_booksum
# from cascade.main.jobs.stream import job_stream
from cascade.main.jobs.mmlu import job_mmlu
from cascade.main.eval_args import eval_args, ArgsType


MODEL_GETTERS = {
    "llama": LlamaForCausalLM,
    "qwen": Qwen2ForCausalLM,
}

CONFIG_GETTERS = {
    "llama": LlamaConfig,
    "qwen": Qwen2Config,
}


def get_model(model_id, **from_pretrained_kwargs):
    keys = list(MODEL_GETTERS.keys())
    key_idx = [1 if k in model_id.lower() else 0 for k in keys].index(1)
    key = keys[key_idx]

    model = MODEL_GETTERS[key].from_pretrained(model_id, **from_pretrained_kwargs)
    return model


def get_config(model_id):
    keys = list(CONFIG_GETTERS.keys())
    key_idx = [1 if k in model_id.lower() else 0 for k in keys].index(1)
    key = keys[key_idx]

    return CONFIG_GETTERS[key].from_pretrained(model_id)


def load_vllm_model(args: ArgsType):
    from vllm import LLM

    device = 'cuda:0'
    MODELS = {
        'vllm_llama32k': 'togethercomputer/LLaMA-2-7B-32K',
        'vllm_llama128k': 'NousResearch/Yarn-Llama-2-7b-128k',
        'vllm_llama13b_128k': 'NousResearch/Yarn-Llama-2-13b-128k',
        'vllm_llama100k': 'Yukang/Llama-2-7b-longlora-100k-ft',
        'vllm_llama32k_instruct': 'togethercomputer/Llama-2-7B-32K-Instruct',
        'vllm_llama1b': 'princeton-nlp/Sheared-LLaMA-1.3B',
        'vllm_llama7b': 'meta-llama/Llama-2-7b-hf',
        'vllm_llama13b': 'meta-llama/Llama-2-13b-hf',
        'vllm_qwen7b': 'Qwen/Qwen1.5-7B-Chat-GPTQ-Int4',
        'vllm_qwen14b': 'Qwen/Qwen1.5-14B-Chat',
        'vllm_qwen0.5b': 'Qwen/Qwen1.5-0.5B-Chat',
        'vllm_pythia70m': 'EleutherAI/pythia-70m',
        'vllm_yi6b': '01-ai/Yi-6B-200K',
        'vllm_yi34b': 'brucethemoose/Yi-34B-200K-RPMerge',
        'vllm_mixtral8x7b': 'TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ',
    }
    assert args.model in MODELS
    assert args.job in ['stream']
    model_id = MODELS[args.model]

    assert args.checkpoint is None

    seq_len = args.stride
    # seq_len = 10600
    model = LLM(
        model_id,
        max_context_len_to_capture=seq_len,
        max_num_seqs=args.batch_size,
        max_model_len=seq_len,
        swap_space=0,
        kv_cache_dtype='fp8_e5m2',
        dtype='half',
        gpu_memory_utilization=0.8,
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=os.environ.get('FORCE_EAGER', '0') == '1',
        trust_remote_code=True,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer, device


def get_dtype(model_name, use_fp32=False):
    if use_fp32:
        return torch.float

    if "llama" in model_name.lower():
        return torch.float16
    elif "qwen" in model_name.lower():
        return torch.float16
    else:
        raise ValueError(f"unknown dtype for model: {model_name}")


def get_injection_policy(model_id):
    if "llama" in model_id.lower():
        return {
            LlamaDecoderLayer: (
                'mlp.down_proj',
                'self_attn.o_proj',
            )
        }
    elif "qwen" in model_id.lower():
        return {
            Qwen2DecoderLayer: (
                'mlp.down_proj',
                'self_attn.o_proj',
            ),
        }
    else:
        raise ValueError()


def model_hash(model):
    import hashlib
    flt = hashlib.shake_256()
    for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
        flt.update(name.encode())
        flt.update(param.data.view(-1)[:min(16, param.data.numel())].cpu().numpy().tobytes())
    return flt.hexdigest(16)


PATH = "/d1/dataset/llama/models/llama_v3.1/"
MODELS = {
    'llama3.1-8b-instruct': os.path.join(PATH, "Meta-Llama-3.1-8B-Instruct"),
    'llama3.1-8b': os.path.join(PATH, "Meta-Llama-3.1-8B"),
    'llama3.1-70b': os.path.join(PATH, "Meta-Llama-3.1-70B"),
    'llama3.1-70b-instruct': os.path.join(PATH, "Meta-Llama-3.1-70B-Instruct"),
    'llama3.1-70b-instruct-gptq-int4': "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4",
    'llama7b': 'togethercomputer/LLaMA-2-7B-32K',
    'llama13b': 'meta-llama/Llama-2-13b-hf',
    'llama13b_32k': 'Yukang/Llama-2-13b-longlora-32k-ft',
    'llama7b-chat': '/d1/dataset/llama/models/llama_v2/llama-2-7b-chat-hf',
    "llama2-7b-chat-32k": "togethercomputer/Llama-2-7B-32K-Instruct",
    'qwen14b': 'Qwen/Qwen1.5-14B',
    'qwen7b': 'Qwen/Qwen1.5-7B',
    'qwen7b-chat': 'Qwen/Qwen1.5-7B-Chat',
    "qwen2-14b-chat-32k": "Qwen/Qwen1.5-14B-Chat",
    "qwen2-7b-chat-32k": "Qwen/Qwen1.5-7B-Chat",
    "qwen2-7b-instruct": "Qwen/Qwen2-7B-Instruct",
    "qwen2-7b": "Qwen/Qwen2-7B",
    'qwen0.5b': 'Qwen/Qwen1.5-0.5B',
    'llama1.3b': 'princeton-nlp/Sheared-LLaMA-1.3B',
    'llama3-8b-instruct':
    "/d1/dataset/llama/models/llama_v3/Meta-Llama-3-8B-Instruct",
    'llama3-8b': 'meta-llama/Meta-Llama-3-8B',
    'llama3-70b-instruct':
    "/d1/dataset/llama/models/llama_v3/Meta-Llama-3-70B-Instruct",
    'llama2-70b': "/d1/dataset/llama/models/llama_v2/llama-2-70b",
}


def load_model(args):
    if args.model.startswith('vllm'):
        return load_vllm_model(args)

    device = 'cuda:0'

    assert args.model in MODELS, MODELS.keys()
    model_id = MODELS[args.model]

    args.local_rank = int(os.getenv('LOCAL_RANK', '0'))
    args.world_size = int(os.getenv('WORLD_SIZE', '1'))

    config = get_config(model_id)

    config._attn_implementation = config.attn_implementation = 'eager'
    if args.method in ["vanilla", "snapkv", "bigbird", "minference-cascade"]:
        config._attn_implementation = config.attn_implementation = 'flash_attention_2'

    if args.job == "latency":
        config.max_position_embeddings = 2 ** 19

    config._batch_size = args.batch_size
    config._sinks = args.sinks
    config._cascades = args.cascades
    config._window = args.window
    config.world_size = args.world_size
    config._cascade_func = args.cascade_func
    config._head_reduction = args.head_reduction
    config._method = args.method
    config._cascade_stride = args.cascade_stride
    config._homogeneous_heads = args.homogeneous_heads
    config._do_og_pos = args.do_og_pos

    print(f"{config=}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    if args.method == "h2o":
        # sinks and window args from cli args are fit into h2o setting within laod function
        from cascade.models.h2o import load
        model, _ = load(model_id, heavy_hitter=True, args=args)
    elif args.method == "minference-cascade":
        config._cascade_stride = 65536 + 32768 + 16384 - args.sinks

        path = MODELS[args.model]
        model = transformers.models.llama.modeling_llama.LlamaForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            device_map="cpu",
        )

        minference_name = args.model
        if "llama" in minference_name.lower():
            minference_name = "meta-llama/" + path.split("/")[-1]
        elif "qwen" in minference_name.lower():
            minference_name = "Qwen/" + path.split("/")[-1]
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

    elif args.method == "minference":
        path = MODELS[args.model]
        model = transformers.models.llama.modeling_llama.LlamaForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.float16,
            device_map="cpu",
        )

        minference_name = args.model
        if "llama" in minference_name.lower():
            minference_name = "meta-llama/" + path.split("/")[-1]
        elif "qwen" in minference_name.lower():
            minference_name = "Qwen/" + path.split("/")[-1]
        else:
            raise NotImplementedError("model not implemented for minference")

        model.minference_name = minference_name

        # Patch MInference Module,
        # If you use the local path, please use the model_name from HF when initializing MInference.
        minference_patch = MInference(
            attn_type="minference",
            model_name=minference_name,
            use_cascade=False,
        )

        model = minference_patch(model)
        model = model.cuda()

    elif args.method == "snapkv":
        if "llama" not in args.model:
            raise ValueError("SnapKV is only implemented for llama models")

        replace_llama()
        model = transformers.models.llama.modeling_llama.LlamaForCausalLM.from_pretrained(
            MODELS[args.model],
            config=config,
            device_map={"": device},
            torch_dtype=torch.float16,
        )

    elif "70b" not in model_id.lower():
        args.infer_dtype = get_dtype(model_id, use_fp32=args.use_fp32)
        from_pretrained_kwargs = dict(
            config=config,
            device_map={"": device},
            torch_dtype=args.infer_dtype,
        )
        model = get_model(model_id, **from_pretrained_kwargs)

    else:
        assert "gptq" in model_id.lower()
        args.infer_dtype = get_dtype(model_id, use_fp32=args.use_fp32)
        from_pretrained_kwargs = dict(
            config=config,
            device_map={"": device},
            torch_dtype=args.infer_dtype,
        )
        model = get_model(model_id, **from_pretrained_kwargs)

    if args.method == "sink":
        model = sample_monkeypatch(model)

    if args.lora_r > 0 and args.checkpoint is not None:
        print("LoRA init")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True,
            r=args.lora_r,
            lora_alpha=args.lora_r // 2,
            lora_dropout=0.0,
            target_modules=[
                'q_proj',
                'k_proj',
                'v_proj',
                'o_proj',
                'gate_proj',
                'up_proj',
                'down_proj',
                # 'input_layernorm', 'post_attention_layernorm'
            ],
            modules_to_save=[
                'input_layernorm',
                'post_attention_layernorm',
            ])

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        state_dict = torch.load(args.checkpoint,
                                map_location='cpu')['state_dict']
        keys = list(state_dict.keys())
        for key in keys:
            x = state_dict[key]
            state_dict[key.strip('model.')] = x
            del state_dict[key]

        result = model.load_state_dict(state_dict, strict=True)
        print('load result', result)
        print('lora checkpoint loaded from', args.checkpoint)

    return model, tokenizer, device


def main():
    seed(seed=42)

    args = eval_args()

    assert args.job in [
        'ppl', 'ppl-pg19', 'ppl-wikitext', 'ppl-memory', 'mmlu',
        'passkey', "latency", "booksum", "attn_matrix_plot"
    ]

    model, tokenizer, device = load_model(args)

    if args.job == 'ppl':
        job_ppl(args, model, tokenizer, device)
    elif args.job == 'attn_matrix_plot':
        job_attn_matrix(args, model, tokenizer, device)
    elif args.job == 'latency':
        job_latency(args, model, tokenizer, device)
    elif args.job == 'passkey':
        job_passkey(args, model, tokenizer, device)
    elif args.job == 'ppl-pg19':
        job_ppl_pg19(args, model, tokenizer, device)
    elif args.job == 'ppl-wikitext':
        job_ppl_wikitext2(args, model, tokenizer, device)
    elif args.job == 'booksum':
        job_booksum(args, model, tokenizer, device)
    elif args.job == 'mmlu':
        job_mmlu(args, model, tokenizer, device)
    else:
        raise Exception()


if __name__ == '__main__':
    main()
