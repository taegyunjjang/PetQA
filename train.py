import os
import copy
import json
import re
import torch

from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
cache_dir = "./models"


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

prompt = (
"당신은 반려동물(개, 고양이) 의료 상담 전문 수의사입니다.\n"
"당신의 역할은 개(강아지), 고양이 관련 의료 질문에 대해 유용하고, 완전하며, 전문적인 지식에 기반한 정확한 답변을 하는 것입니다.\n\n"
"### 질문: {question}\n"
"### 답변:"
)

data_path = "data/preprocessed_data.json"
list_data_dict = load_dataset("json", data_files=data_path, split="train")

sources = [
    prompt.format_map(example) for example in list_data_dict
]

targets = [f"{example['answer']}{DEFAULT_EOS_TOKEN}" for example in list_data_dict]

model_name = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # 이중 양자화로 메모리 추가 절약
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto",
    use_cache=False
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token  # padding 토큰 설정

def _tokenize_fn(strings, tokenizer):
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources, targets, tokenizer):
    examples = [s + t for s, t in zip(sources, targets)]
    
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    
    input_ids = examples_tokenized['input_ids']
    labels = copy.deepcopy(input_ids)
    
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
        
    return dict(input_ids=input_ids, labels=labels)

data_dict = preprocess(sources, targets, tokenizer)

tokenized_dataset = dataset.map(
    lambda x: preprocess(x['sources'], x['targets'], tokenizer),
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing dataset",
)

# LoRA 적용
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # 모델 구조에 따라 다름
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 학습
training_args = TrainingArguments(
    output_dir="./exaone-7.8B",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    fp16=True,
    num_train_epochs=1,
    logging_steps=50,
    save_steps=500,
    optim="paged_adamw_8bit",  # LoRA + QLoRA 전용 옵티마이저
    report_to="none",
    learning_rate=2e-4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()

trainer.save_model("./exaone-7.8B-lora/final")
print("모델 학습 및 저장 완료!")

# 학습된 모델 추론 예시
def generate_answer(question, model, tokenizer):
    prompt = f"질문: {question}\n답변:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_output[len(prompt):]  # 프롬프트 부분 제외하고 응답만 반환
    return answer


# 테스트 예시
test_question = "강아지 피뽑고 붕대는 언제쯤 풀 수 있나요?"
print(f"질문: {test_question}")
print(f"답변: {generate_answer(test_question, model, tokenizer)}")