import json
import os
from tqdm import tqdm
from dotenv import load_dotenv
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline


MODEL_MAPPING = {
    "gemma-3-4b": "google/gemma-3-4b-it",
    "exaone-3.5-7.8b": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
}


def load_environment(model_name, shot, use_raw_format):
    suffix = "_raw" if use_raw_format else ""
    cache_dir = "./models"
    
    fintuned_path = f'data/outputs/{model_name}{suffix}'
    test_data_path = f"data/training/test_data{suffix}.json"
    output_path = f"data/eval/output_{model_name}_{shot}_{suffix}.json"
    
    return {
        "cache_dir": cache_dir,
        "fintuned_path": fintuned_path,
        "test_data_path": test_data_path,
        "output_path": output_path
    }
    
def load_model_and_tokenizer(model_name, fintuned_path, cache_dir):
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        fintuned_path,
        cache_dir=cache_dir,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_MAPPING[model_name],
        cache_dir=cache_dir
    )
    text_gen_pipeline = pipeline("text-generation", model=finetuned_model, tokenizer=tokenizer)
    return text_gen_pipeline, tokenizer

def load_test_data(test_data_path):
    with open(test_data_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def load_results(output_path):
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        start_idx = len(results)
        print(f"{start_idx}개까지 처리됨. 이어서 시작")
    else:
        results = []
        start_idx = 0
        print("새로 시작")
    return results, start_idx

def generate_answers(pipe, tokenizer, test_data, results, start_idx, output_path, use_raw_format):
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]

        
    for item in tqdm(test_data[start_idx:], total=len(test_data) - start_idx, desc="답변 생성 중"):
        if use_raw_format:
            question = f"{item['title']}\n\n{item['content']}" if item['content'] else item['title']
        else:
            question = item['question']
            
        messages = [{"role": "user", "content": question}]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        outputs = pipe(
            prompt,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.1,
            top_p=0.1,
            top_k=50,
            repetition_penalty=1.2,
            eos_token_id=stop_token_ids
        )

        generated_answer = outputs[0]['generated_text'][len(prompt):]

        if use_raw_format:
            result = {
                "id": item["id"],
                "title": item["title"],
                "content": item["content"],
                "answer": item["answer"],
                "answer_date": item["answer_date"],
                "generated_answer": generated_answer
            }
        else:
            result = {
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "answer_date": item["answer_date"],
                "generated_answer": generated_answer
            }
            
        results.append(result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"결과가 {output_path}에 저장되었습니다.")
    
def main(model_name, shot, use_raw_format):
    env = load_environment(model_name, shot, use_raw_format)
    pipe, tokenizer = load_model_and_tokenizer(model_name, env["fintuned_path"], env["cache_dir"])
    test_data = load_test_data(env["test_data_path"])
    results, start_idx = load_results(env["output_path"])
    generate_answers(pipe, tokenizer, test_data, results, start_idx, env["output_path"], use_raw_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="답변 생성")
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_MAPPING.keys()))
    parser.add_argument("--shot", type=str, required=True, choices=["0", "1", "3", "6"])
    parser.add_argument("--use_raw_format", action="store_true")
    args = parser.parse_args()
    
    main(args.model_name, args.shot, args.use_raw_format)
