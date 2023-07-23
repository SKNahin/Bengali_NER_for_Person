from utils.helper import *
from utils.preprocessor import *
from utils.trainer import *

from datasets import Dataset as Datasets
from datasets import DatasetDict
from transformers import AutoModelForTokenClassification, TrainingArguments, DataCollatorForTokenClassification


# Defining our device
if torch.cuda.is_available():
    device=torch.device("cuda:0")
    print("Running on the GPU")
    torch.cuda.empty_cache()

else:
    device=torch.device("cpu")
    print("Running on the CPU")

# loading data
train_data = load_json("./data/processed_data/balanced/train.json")
valid_data = load_json("./data/processed_data/balanced/valid.json")
test_data = load_json("./data/processed_data/balanced/test.json")

# making dictionary of our dataset
all_data_dict = DatasetDict({"train": Datasets.from_dict(train_data),"valid":Datasets.from_dict(valid_data),"test":Datasets.from_dict(test_data)})

# Creating the dataset according to the preprocessing function
all_dataset = all_data_dict.map(preprocess_function, batched=True)
all_dataset_back = all_dataset.copy() # Keeping a copy
all_dataset=all_dataset.remove_columns(['sentences','entity','label']) #Dropping unnecessary columns for training

# Preparing the collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


# defining training arguments
training_args = TrainingArguments(
    output_dir="./checkpoints",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    metric_for_best_model="F1",
    num_train_epochs=1,
    weight_decay=0.01,
    include_inputs_for_metrics =True,
    seed = 0,
    save_steps=3500
)


# time to train....
if __name__ == "__main__":
    # now loading the model
    model = AutoModelForTokenClassification.from_pretrained("csebuetnlp/banglabert", num_labels=2)

    # defining the trainer
    trainer = MyTrainer(
        model=model.to(device),
        args=training_args,
        train_dataset=all_dataset["train"],
        eval_dataset=all_dataset["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
