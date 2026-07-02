import json

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\\n" for line in text.split("\\n")]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\\n" for line in text.split("\\n")]
    })

add_md("""# Evaluate Medical Models on General Datasets
Models to evaluate (in order):
1. `google/medgemma-4b-it`
2. `microsoft/llava-med-v1.5-mistral-7b`
3. `FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL`

Datasets: `lmms-lab/VQAv2` (sampled), `lmms-lab/OK-VQA` (sampled)

Note: For 7B models on Kaggle T4, we use 4-bit quantization via bitsandbytes.""")

add_code("""!pip install -q transformers==4.40.1 datasets bitsandbytes accelerate pillow""")

add_code("""import os, json, re, random
import torch
from PIL import Image
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import BitsAndBytesConfig, AutoProcessor, LlavaForConditionalGeneration, AutoModelForImageTextToText

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
print(f'PyTorch: {torch.__version__}')

def to_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == 'RGB' else img.convert('RGB')
""")

add_code("""# ── Dataset Loading (Streaming 1,000 samples) ──────────────────────────────────

print('Streaming VQAv2 validation split...')
vqav2_stream = load_dataset('lmms-lab/VQAv2', split='validation', streaming=True)

def get_vqav2_answer(sample):
    answers = [a['answer'].strip().lower() for a in sample['answers']]
    return max(set(answers), key=answers.count)

def is_vqav2_closed(sample):
    ans = get_vqav2_answer(sample)
    return ans in ('yes', 'no')

closed_samples = []
open_samples = []

for sample in vqav2_stream:
    if len(closed_samples) >= 500 and len(open_samples) >= 500:
        break
        
    if is_vqav2_closed(sample):
        if len(closed_samples) < 500:
            closed_samples.append(sample)
    else:
        if len(open_samples) < 500:
            open_samples.append(sample)

vqav2_test = closed_samples + open_samples
random.seed(42)
random.shuffle(vqav2_test)

print(f'VQAv2 sampled test: {len(vqav2_test)} samples (500 closed, 500 open)')

print('Streaming OK-VQA dataset...')
okvqa_stream = load_dataset('lmms-lab/OK-VQA', split='val2014', streaming=True)
okvqa_test = []
for sample in okvqa_stream:
    if len(okvqa_test) >= 1000:
        break
    okvqa_test.append(sample)

print(f'OK-VQA sampled test: {len(okvqa_test)} samples')
""")

add_code("""# ── Prompt Engineering (v2 protocol) ──────────────────────────────────

def build_prompt_vqav2(question: str, is_closed: bool) -> str:
    prefix = 'Answer the question with yes or no. ' if is_closed else ''
    return (
        f"{prefix}{question} "
        f"You may write out your argument before stating your final very short, "
        f"definitive, and concise answer (if possible, a single word) "
        f"X in the format 'Final Answer: X'"
    )

def build_prompt_okvqa(question: str, is_closed: bool) -> str:
    return (
        f"{question} "
        f"You may write out your argument before stating your final very short, "
        f"definitive, and concise answer (if possible, a single word) "
        f"X in the format 'Final Answer: X'"
    )

def extract_final_answer(text: str) -> str:
    match = re.search(r'[Ff]inal\\s+[Aa]nswer\\s*:\\s*(.+)', text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        answer = re.sub(r'[\\*\"\\']+'  , '', answer).strip()
        answer = answer.split('\\n')[0].strip()
        return answer
    return re.split(r'(?<=[.!?])\\s', text)[0].strip()
""")

add_code("""# ── Generic Runner ──────────────────────────────────

def run_dataset(
    model, processor, device, samples,
    get_image_fn, get_question_fn, get_answer_fn, get_is_closed_fn,
    build_prompt_fn, dataset_name, model_name,
    output_dir='./outputs', is_qwen=False
):
    os.makedirs(output_dir, exist_ok=True)
    safe_model = model_name.replace('/', '_')
    out_path = os.path.join(output_dir, f'{safe_model}__{dataset_name}_v2.jsonl')
    
    completed = set()
    if os.path.exists(out_path):
        with open(out_path, 'r') as f:
            for line in f:
                r = json.loads(line)
                completed.add(r['idx'])
        print(f'Resuming: {len(completed)} samples already done for {model_name} on {dataset_name}.')

    errors = 0
    f_out = open(out_path, 'a')
    
    for i, sample in enumerate(tqdm(samples, desc=f'{model_name} | {dataset_name}')):
        if i in completed:
            continue
            
        try:
            image = to_rgb(get_image_fn(sample))
            question = get_question_fn(sample)
            answer = get_answer_fn(sample)
            is_closed = get_is_closed_fn(sample)
            
            prompt_text = build_prompt_fn(question, is_closed)
            
            if is_qwen:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text],
                    images=[image],
                    padding=True,
                    return_tensors="pt",
                ).to(device)
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt_text},
                        ]
                    }
                ]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=text, images=image, return_tensors='pt'
                ).to(device)
            
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs, max_new_tokens=100, do_sample=False
                )
            
            # Remove input length from output
            if is_qwen:
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                raw = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )[0].strip()
            else:
                input_len = inputs['input_ids'].shape[-1]
                raw = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
                
            prediction = extract_final_answer(raw)
            
            record = {
                'idx': i, 'question': question,
                'ground_truth': answer, 'prediction': prediction,
                'raw_output': raw, 'is_closed': is_closed,
                'model': model_name, 'dataset': dataset_name,
            }
        except Exception as e:
            errors += 1
            record = {
                'idx': i, 'question': '', 'ground_truth': '',
                'prediction': '', 'raw_output': '',
                'is_closed': False, 'model': model_name, 'dataset': dataset_name,
                'error': str(e),
            }
            
        f_out.write(json.dumps(record) + '\\n')
        f_out.flush()
        
    f_out.close()
    print(f'Done. Output saved to {out_path}')
""")

add_md("## 1. MedGemma Inference (No Quantization Needed for 4B on 16GB VRAM)")

add_code("""medgemma_model_id = "google/medgemma-4b-it"
medgemma_processor = AutoProcessor.from_pretrained(medgemma_model_id)
medgemma_model = AutoModelForImageTextToText.from_pretrained(
    medgemma_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
""")

add_code("""# Run MedGemma on VQAv2
run_dataset(
    model=medgemma_model, processor=medgemma_processor, device=device,
    samples=vqav2_test,
    get_image_fn=lambda x: x['image'],
    get_question_fn=lambda x: x['question'],
    get_answer_fn=get_vqav2_answer,
    get_is_closed_fn=is_vqav2_closed,
    build_prompt_fn=build_prompt_vqav2,
    dataset_name="vqav2",
    model_name=medgemma_model_id,
    is_qwen=False
)

# Run MedGemma on OK-VQA
run_dataset(
    model=medgemma_model, processor=medgemma_processor, device=device,
    samples=okvqa_test,
    get_image_fn=lambda x: x['image'],
    get_question_fn=lambda x: x['question'],
    get_answer_fn=lambda x: max(set([a.strip().lower() for a in x['answers']]), key=[a.strip().lower() for a in x['answers']].count),
    get_is_closed_fn=lambda x: False,
    build_prompt_fn=build_prompt_okvqa,
    dataset_name="okvqa",
    model_name=medgemma_model_id,
    is_qwen=False
)

# Free memory
import gc
del medgemma_model
del medgemma_processor
gc.collect()
torch.cuda.empty_cache()
""")

add_md("## 2. LLaVA-Med Inference (with 4-bit quantization)")

add_code("""quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

llava_model_id = "microsoft/llava-med-v1.5-mistral-7b"
llava_processor = AutoProcessor.from_pretrained(llava_model_id)
llava_model = LlavaForConditionalGeneration.from_pretrained(
    llava_model_id,
    device_map="auto",
    quantization_config=quantization_config
)
""")

add_code("""# Run LLaVA-Med on VQAv2
run_dataset(
    model=llava_model, processor=llava_processor, device=device,
    samples=vqav2_test,
    get_image_fn=lambda x: x['image'],
    get_question_fn=lambda x: x['question'],
    get_answer_fn=get_vqav2_answer,
    get_is_closed_fn=is_vqav2_closed,
    build_prompt_fn=build_prompt_vqav2,
    dataset_name="vqav2",
    model_name=llava_model_id,
    is_qwen=False
)

# Run LLaVA-Med on OK-VQA
run_dataset(
    model=llava_model, processor=llava_processor, device=device,
    samples=okvqa_test,
    get_image_fn=lambda x: x['image'],
    get_question_fn=lambda x: x['question'],
    get_answer_fn=lambda x: max(set([a.strip().lower() for a in x['answers']]), key=[a.strip().lower() for a in x['answers']].count),
    get_is_closed_fn=lambda x: False,
    build_prompt_fn=build_prompt_okvqa,
    dataset_name="okvqa",
    model_name=llava_model_id,
    is_qwen=False
)

# Free memory
del llava_model
del llava_processor
gc.collect()
torch.cuda.empty_cache()
""")

add_md("## 3. HuatuoGPT Inference (with 4-bit quantization)")

add_code("""from transformers import Qwen2_5_VLForConditionalGeneration

huatuo_model_id = "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"
huatuo_processor = AutoProcessor.from_pretrained(huatuo_model_id)
huatuo_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    huatuo_model_id,
    device_map="auto",
    quantization_config=quantization_config
)
""")

add_code("""# Run HuatuoGPT on VQAv2
run_dataset(
    model=huatuo_model, processor=huatuo_processor, device=device,
    samples=vqav2_test,
    get_image_fn=lambda x: x['image'],
    get_question_fn=lambda x: x['question'],
    get_answer_fn=get_vqav2_answer,
    get_is_closed_fn=is_vqav2_closed,
    build_prompt_fn=build_prompt_vqav2,
    dataset_name="vqav2",
    model_name=huatuo_model_id,
    is_qwen=True
)

# Run HuatuoGPT on OK-VQA
run_dataset(
    model=huatuo_model, processor=huatuo_processor, device=device,
    samples=okvqa_test,
    get_image_fn=lambda x: x['image'],
    get_question_fn=lambda x: x['question'],
    get_answer_fn=lambda x: max(set([a.strip().lower() for a in x['answers']]), key=[a.strip().lower() for a in x['answers']].count),
    get_is_closed_fn=lambda x: False,
    build_prompt_fn=build_prompt_okvqa,
    dataset_name="okvqa",
    model_name=huatuo_model_id,
    is_qwen=True
)
""")

notebook = {
    "cells": cells,
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('notebooks/07_medical_on_general_v2.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully!")
