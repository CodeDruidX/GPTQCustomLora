

# GPTQLoRA: Efficient Finetuning of Quantized LLMs with GPTQ

QLoRA uses [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) for quantization

## License and Intended Use
We release the resources associated with QLoRA finetuning in this repository under MIT license.

## Installation
To load models in 4bits with transformers and bitsandbytes, you have to install accelerate and transformers from source and make sure you have the latest version of the bitsandbytes library (0.39.0). You can achieve the above with the following commands:
```bash
git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
pip install .[triton]
cd ..
pip install -r requirements.txt
pip install bitsandbytes
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/qwopqwop200/peft.git
pip install git+https://github.com/huggingface/accelerate.git
```

## Getting Started
The `qlora.py` code is a starting point for finetuning and inference on various datasets.
Basic command for finetuning a baseline model on the Alpaca dataset:
```bash
python qlora.py --model_path <path>
```

For models larger than 13B, we recommend adjusting the learning rate:
```bash
python qlora.py –learning_rate 0.0001 --model_path <path>
```

## Quantization
Quantization parameters are controlled from the `BitsandbytesConfig` ([see HF documenation](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig)) as follows:
- Loading in 4 bits is activated through `load_in_4bit`
- The datatype used for the linear layer computations with `bnb_4bit_compute_dtype`
- Nested quantization is activated through `bnb_4bit_use_double_quant`
- The datatype used for qunatization is specified with `bnb_4bit_quant_type`. Note that there are two supported quantization datatypes `fp4` (four bit float) and `nf4` (normal four bit float). The latter is theoretically optimal for normally distributed weights and we recommend using `nf4`.

```python
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path='/name/or/path/to/your/model',
        load_in_4bit=True,
        device_map='auto',
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        ),
    )
```

## Paged Optimizer
You can access the paged optimizer with the argument `--optim paged_adamw_32bit`

## Known Issues and Limitations
Here a list of known issues and bugs. If your issue is not reported here, please open a new issue and describe the problem.

1. Resuming a LoRA training run with the Trainer currently runs on an error
2. Make sure that `tokenizer.bos_token_id = 1` to avoid generation issues.

## Acknoledgements
We thank the Huggingface team, in particular Younes Belkada, for their support integrating QLoRA with PEFT and transformers libraries.

This repo builds on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) and [LMSYS FastChat](https://github.com/lm-sys/FastChat) repos.
