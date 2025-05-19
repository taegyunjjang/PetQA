import pandas as pd


file_path = "data/cleaned_data.json"
df = pd.read_json(file_path)
df['title'] = df['제목']
df['content'] = df['본문']
df['answer'] = df['답변']
df['preprocessed_question'] = df['cleaned_question']
df['preprocessed_answer'] = df['cleaned_answer']

df = df[['id', 'title', 'content', 'answer', 'preprocessed_question', 'preprocessed_answer', 'answer_date']]

df = df.sample(frac=1, random_state=42)

val_size = 2000
test_size = 2000

val_df = df.iloc[:val_size]
test_df = df.iloc[val_size:val_size+test_size]
train_df = df.iloc[val_size+test_size:]

train_df.to_json("data/training/train.json", orient="records", force_ascii=False, indent=4)
val_df.to_json("data/training/validation.json", orient="records", force_ascii=False, indent=4)
test_df.to_json("data/training/test.json", orient="records", force_ascii=False, indent=4)
