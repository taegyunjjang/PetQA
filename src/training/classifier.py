import wandb
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
    DataCollatorWithPadding
)
from dotenv import load_dotenv
load_dotenv()
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from utils.utils import (
    setup_logging, load_environment
)


def prepare_dataset(data_files, logger):
    logger.info("Loading datasets...")
    dataset = load_dataset("json", data_files=data_files)
    
    logger.info("Preprocessing train and validation splits...")
    label2id = {"expert": 0, "nonexpert": 1}
    id2label = {0: "expert", 1: "nonexpert"}
    def preprocess_function(examples):
        return {
            "questions": examples["preprocessed_question"],
            "labels": [label2id[label] for label in examples["answer_type"]]
        }
    preprocessed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    return preprocessed_dataset, label2id, id2label

def load_model_tokenizer(model_name, label2id, id2label, logger):
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def tokenize_dataset(dataset, tokenizer, logger, max_length=512):
    logger.info("Tokenizing datasets...")
    def tokenize_function(examples):
        return tokenizer(
            examples["questions"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["questions"]
    )
    return tokenized_datasets

def train_classifier(tokenized_datasets, model, tokenizer, output_dir, logger):
    accuracy_metric = evaluate.load("accuracy")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
        logging_steps=100,
        report_to="wandb",
        save_total_limit=3,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed!")
    
    best_model_dir = f"{output_dir}/best_model"
    trainer.save_model(best_model_dir)
    logger.info(f"Best model saved to {best_model_dir}")
    return trainer, best_model_dir

def test_classifier(model_path, logger):
    pipe = pipeline("text-classification", model=model_path)
    
    # E, NE, NE, E
    test_questions = [
        "강아지가 푸들인데 지금 6개월 되었습니다. 다른 이빨은 다 유치가 빠졌는데 송곳니만 아직 안 빠졌습니다. 흔들림도 전혀 없고, 영구치가 바로 옆에 또 나고 있는데 어떻게 해야 하나요?",
        "약 1달 전에 길고양이를 데려와 키우고 있습니다. 어미가 없어 보여 데려왔는데, 적응할수록 입질이 심해집니다. 처음에는 약해서 괜찮았지만, 이제는 아파서 밀쳐내면 더 심해집니다. 특히 밤에 잘 때 심해서 잠을 못 자 이층침대를 사려고 하는데, 아빠는 몇 개월 있으면 소용없어질 거라고 합니다. 성묘가 되면 입질이 덜할까요?",
        "이모가 키우는 강아지에게 손톱 주위 살이 얕게 찢어지는 상처를 입었습니다. 피가 조금 나서 수돗물로 씻고 후시딘 연고를 발랐는데 열감이 느껴지고 약간 부어올랐습니다. 강아지는 광견병 주사를 맞았고 산책 후 흙과 풀에 많이 다닌 직후에 물린 건인데, 이 경우 근처 내과에 내원하는 것이 좋을까요?",
        "생리 중인 포메라니안 강아지가 생식기를 자꾸 핥으려고 하고, 발도 물어뜯습니다. 항상 앉아 있고 걸어 다닐 때도 뒷다리를 어색하게 사용하며 엉덩이를 끌고 다닙니다. 항문낭은 짜도 나오지 않고, 움직이지 않고 한 곳에 배와 생식기 쪽을 내민 채 앉아 있습니다. 이러한 증상이 생리 중 나타날 수 있는 정상적인 증상인가요?"
    ]
    
    logger.info("Testing classifier...")
    for question in test_questions:
        result = pipe(question)
        logger.info(f"Question: {question}")
        logger.info(f"Prediction: {result[0]['label']} (confidence: {result[0]['score']:.4f})")
        logger.info("-" * 50)

if __name__ == "__main__":
    env = load_environment()
    logger = setup_logging()
    
    wandb_run_name = f"question_type_classifier"
    output_dir = os.path.join(env["checkpoint_dir"], wandb_run_name)
    os.environ["WANDB_DIR"] = output_dir
    wandb.init(
        project="PetQA",
        entity="petqa",
        name=wandb_run_name
    )
    
    model_name = "klue/roberta-large"
    dataset, label2id, id2label = prepare_dataset(env["data_files"], logger)
    model, tokenizer = load_model_tokenizer(model_name, label2id, id2label, logger)
    tokenized_datasets = tokenize_dataset(dataset, tokenizer, logger)
    trainer, best_model_dir = train_classifier(tokenized_datasets, model, tokenizer, output_dir, logger)
    test_classifier(best_model_dir, logger)
    wandb.finish()
    logger.info("Classifier training completed!")