import os
os.environ['VLLM_USE_V1'] = '0'
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from vllm import LLM, SamplingParams
from utils.utils import (
    MODEL_MAPPING, setup_logging, save_json, load_json,
    load_prompt, load_results, load_environment
)
from etc.blocker_numpy import blocker


def load_model_and_tokenizer(model_path):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def get_prompts(data_to_process, tokenizer, args, env):
    prompts = []
    images = []
    base_user_prompt = load_prompt(env["zeroshot_prompt_path"])
    for item in data_to_process:
        user_prompt = base_user_prompt.format(question=item["question"])
        
        if args.use_image_data:
            image_id = item["image"]
            image = Image.open(os.path.join(env["image_dir"], f"{image_id}.jpg"))
            images.append({"image": image})
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt},
                    ]
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                    ]
                }
            ]
            
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompts.append(prompt)
    return prompts, images


def generate_answers(
    llm, tokenizer, test_data, results, start_idx,
    env, args, output_path, batch_size
):
    all_prompts, all_images = get_prompts(test_data[start_idx:], tokenizer, args, env)

    total_prompts = len(all_prompts)
    num_batches = (total_prompts + batch_size - 1) // batch_size

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=512,
        seed=42,
    )

    if args.model_name == "qwen-2.5-vl-7b":
        def _logits_processor(input_ids, logits):
            return blocker(tokenizer, input_ids, logits)
        sampling_params.logits_processors = [_logits_processor]

    for batch_idx in tqdm(range(0, total_prompts, batch_size), total=num_batches, desc="Batch Generation"):
        batch_prompts = all_prompts[batch_idx:batch_idx + batch_size]
        batch_images = all_images[batch_idx:batch_idx + batch_size]
        batch_data = test_data[start_idx + batch_idx:start_idx + batch_idx + batch_size]

        if args.use_image_data:
            inputs = [
                    {"prompt": p, "multi_modal_data": img}
                    for p, img in zip(batch_prompts, batch_images)
                ]
        else:
            inputs = [
                {"prompt": p}
                for p in batch_prompts
            ]

        outputs = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
        generated_texts = [o.outputs[0].text.strip() for o in outputs]

        for k, item in enumerate(batch_data):
            item["generated_answer"] = generated_texts[k]
            results.append(item)

        save_json(results, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="멀티모달 답변 생성")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="qwen-2.5-vl-7b")
    parser.add_argument("--shot", type=int, choices=[0, 1, 3, 6], default=0)
    parser.add_argument("--use_image_data", action="store_true")
    args = parser.parse_args()

    env = load_environment()
    logger = setup_logging()

    model_path = MODEL_MAPPING[args.model_name]
    
    endpoint = f"{args.model_name}_{args.shot}-shot_w_image.json" if args.use_image_data else f"{args.model_name}_{args.shot}-shot.json"
    output_path = os.path.join(env["generated_answers_dir"], endpoint)

    llm, tokenizer = load_model_and_tokenizer(model_path)
    results, start_idx = load_results(output_path)

    test_data = load_json(env["multimodal_data_path"])
    
    batch_size = 200
    generate_answers(
        llm, tokenizer, test_data, results, start_idx, 
        env, args, output_path, batch_size)

    logger.info("배치 답변 생성 및 저장 완료")