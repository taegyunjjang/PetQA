import pandas as pd
from tqdm import tqdm
import argparse
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from utils.utils import (
    MODEL_MAPPING, setup_logging, save_json, load_json,
    load_prompt, load_results, load_environment
)

from dotenv import load_dotenv
import openai
# import anthropic
# import google.generativeai as genai


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
# anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
# genai.configure(api_key=GOOGLE_API_KEY)


def parse_model_name(model_name):
    if model_name.startswith("gpt"):
        return "gpt"
    elif model_name.startswith("claude"):
        return "claude"
    elif model_name.startswith("gemini"):
        return "gemini"
  
def load_fewshot_examples(env):
    fewshot_df = pd.read_json(env["fewshot_examples_path"])
    fewshot_map = {row["id"]: row for _, row in fewshot_df.iterrows()}
    return fewshot_map

def build_fewshot_examples(sample, shot, input_format):
    examples = ""
    for i in range(int(shot)):
        title = sample["similar_questions"][i]["title"]
        content = sample["similar_questions"][i]["content"]
        question = sample["similar_questions"][i]["preprocessed_question"]
        answer = sample["similar_questions"][i]["preprocessed_answer"]
        
        if input_format == "raw":
            examples += f"제목: {title}\n본문: {content}\n답변: {answer}\n\n"
        else:
            examples += f"질문: {question}\n답변: {answer}\n\n"
    return examples.strip()

def get_prompts(env, item, shot, input_format):
    # system prompt
    if shot == "0":
        system_prompt = load_prompt(env["system_zeroshot_prompt_path"])
    else:
        fewshot_map = load_fewshot_examples(env)
        id = item["id"]
        sample = fewshot_map.get(id)
        fewshot_examples = build_fewshot_examples(sample, shot, input_format)
        base_system_prompt = load_prompt(env["system_fewshot_prompt_path"])
        system_prompt = base_system_prompt.format(fewshot_examples=fewshot_examples)
    
    # user prompt
    if input_format == "raw":
        base_user_prompt = load_prompt(env["user_raw_input_prompt_path"])
        user_prompt = base_user_prompt.format(title=item["title"], content=item["content"])
    else:
        base_user_prompt = load_prompt(env["user_processed_input_prompt_path"])
        user_prompt = base_user_prompt.format(question=item["preprocessed_question"])
        
    return system_prompt, user_prompt

def get_model_processor(model_name):
    model_family = parse_model_name(model_name)
    if model_family == "gpt":
        return lambda *args: generate_gpt_answer(openai_client, model_name, *args)
    # elif model_family == "claude":
    #     return lambda *args: generate_claude_answer(anthropic_client, model_name, *args)
    # elif model_family == "gemini":
    #     return lambda *args: generate_gemini_answer(model_name, *args)

def generate_gpt_answer(client, model_name, *args):
    system_prompt, user_prompt = args
    response = client.chat.completions.create(
        model=MODEL_MAPPING[model_name],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}],
        max_tokens=512,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# def generate_claude_answer(client, model_name, *args):
#     system_prompt, user_prompt = args
#     message = client.messages.create(
#         model=MODEL_MAPPING[model_name],
#         temperature=0,
#         max_tokens=512,
#         system=system_prompt,
#         messages=[
#             {"role": "user", "content": user_prompt}
#         ]
#     )
#     return message.content[0].text

# def generate_gemini_answer(model_name, *args):
#     system_prompt, user_prompt = args
#     model = genai.GenerativeModel(
#         model_name=MODEL_MAPPING[model_name], 
#         system_instruction=system_prompt
#     )
    
#     generation_config = {
#         "temperature": 0, 
#         "max_output_tokens": 512,
#     }
    
#     response = model.generate_content(
#         contents=[
#             {"role": "user", "parts": [user_prompt]}
#         ],
#         generation_config=generation_config
#     )
#     return response.text   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="답변 생성")
    parser.add_argument("--model_name", type=str, choices=list(MODEL_MAPPING.keys()), default="gpt-4o-mini")
    parser.add_argument("--shot", type=str, choices=["0", "1", "3", "6"], default="0")
    parser.add_argument("--input_format", choices=["preprocessed", "raw"], default="preprocessed")
    args = parser.parse_args()
    
    env = load_environment()
    output_path = os.path.join(env["generated_answers_dir"],
                               f"output_{args.model_name}_{args.shot}_{args.input_format}.json")
    logger = setup_logging()
    logger.info(f"MODEL NAME: {args.model_name}")
    logger.info(f"SHOT: {args.shot}")
    logger.info(f"INPUT FORMAT: {args.input_format}")
    logger.info(f"OUTPUT PATH: {output_path}")
    
    results, start_idx = load_results(output_path)
    model_processor = get_model_processor(args.model_name)
    
    test_data = load_json(env["test_data_path"])
    data_to_process = test_data[start_idx:]
    for item in tqdm(data_to_process, total=len(data_to_process), desc="답변 생성 중"):
        system_prompt, user_prompt = get_prompts(env, item, args.shot, args.input_format)
        generated_answer = model_processor(system_prompt, user_prompt)
        item['generated_answer'] = generated_answer
        results.append(item)
        save_json(results, output_path)
        
    logger.info("답변 생성 완료")
    