import time
import os
import pandas as pd
from tqdm import tqdm
import json

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

import openai
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

cache_dir = "./models"

with open("prompt/generating_answer_system.txt") as file:
    system_prompt = file.read()
    
with open("prompt/generating_answer_0shot_gpt_user.txt") as file:
    gpt_user_prompt = file.read()
    
with open("prompt/generating_answer_0shot_gpt_raw_user.txt") as file:
    gpt_raw_user_prompt = file.read()
    
with open("prompt/generating_answer_0shot_exaone_user.txt") as file:
    exaone_user_prompt = file.read()


def generate_gpt_answer(title, text):
    user_prompt = gpt_raw_user_prompt.replace("{title}", title).replace("{text}", text)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}],
        seed=42
    )

    return response.choices[0].message.content.strip()

def generate_exaone_answer(question, model, tokenizer):
    user_prompt = exaone_user_prompt.replace("{question}", question)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    output = model.generate(
        input_ids.to("cuda"),
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=1024,
        do_sample=False
    )
    
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=False)

    start_token = "[|assistant|]"
    end_token = "[|endofturn|]"
    
    # 답변 추출
    if start_token in decoded_output:
        answer = decoded_output.split(start_token, 1)[1].strip()
        if end_token in answer:
            return answer.split(end_token, 1)[0].strip()
        else:
            return answer.strip()
    else:
        return decoded_output.strip()

def main(df, model_id):
    df_len = len(df)
    generated_data = []
    start_idx = 0
    output_path = f"data/output_{model_id}_0shot_raw_text.json"
    
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            generated_data = json.load(f)
        start_idx = len(generated_data)
        print(f"{start_idx}개까지 처리됨. 이어서 시작")
    else:
        print("새로 시작")
    
    if model_id == "gpt-4o-mini":
        for i, row in tqdm(df.iloc[start_idx:].iterrows(), total=df_len - start_idx, desc="답변 생성 중"):
            df.loc[i, 'generated_answer'] = generate_gpt_answer(row['제목'], row['본문'])
            generated_data.append(df.loc[i].to_dict())
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(generated_data, f, ensure_ascii=False, indent=4)
    else:
        model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        for i, row in tqdm(df.iloc[start_idx:].iterrows(), total=df_len - start_idx, desc="답변 생성 중"):
            df.loc[i, 'generated_answer'] = generate_exaone_answer(row['question'], model, tokenizer)
            generated_data.append(df.loc[i].to_dict())
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(generated_data, f, ensure_ascii=False, indent=4)
        
    print(f"답변 생성 완료: {output_path}")


if __name__ == "__main__":
    # df = pd.read_json("data/test_data.json")
    df = pd.read_json("data/cleaned_data.json")
    main(df, model_id="gpt-4o-mini")
    