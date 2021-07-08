import re
import json
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM

with open('schiele.txt') as f:
    data_lines = f.readlines()

def build_text_files(data_lines, dest_path):
    f = open(dest_path, 'w')
    data = ''
    for line in data_lines:
        summary = str(line).strip()
        #summary = re.sub(r'\s+', ' ', summary)
        data += summary + '\n'
    f.write(data)

train, test = train_test_split(data_lines, test_size=0.15)

build_text_files(train, 'train_dataset.txt')
build_text_files(test, 'test_dataset.txt')

print(f'Train dataset length: {len(train)}')
print(f'Test dataset length: {len(test)}')

tokenizer = AutoTokenizer.from_pretrained('anonymous-german-nlp/german-gpt2')

train_path = 'train_dataset.txt'
test_path = 'test_dataset.txt'

def load_dataset(train_path, text_path, tokenizer):
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128
    )

    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    return train_dataset, test_dataset, data_collator

train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)

model = AutoModelForCausalLM.from_pretrained('anonymous-german-nlp/german-gpt2')

training_args = TrainingArguments(
    output_dir='./gpt2-schiele',
    overwrite_output_dir=True,
    num_train_epochs=30,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_steps=400,
    save_steps=800,
    warmup_steps=500,
    prediction_loss_only=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()
trainer.save_model()