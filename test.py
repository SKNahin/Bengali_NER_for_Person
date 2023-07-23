from train import *
import pandas as pd
from transformers import BertConfig

# loading a pretrained model
model = AutoModelForTokenClassification.from_pretrained("./test_weight")

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

# getting all the prediction
out = trainer.predict(all_dataset["test"])
predictions, label_ids ,metric_= out

all_sentences = all_dataset_back["test"]["sentences"]
all_inputs = np.array(all_dataset_back["test"]["input_ids"])
all_labels = np.array(all_dataset_back["test"]["labels"])
all_masks = (all_inputs!=0)*(all_inputs!=2)*(all_inputs!=3)*1 # excluding start, end and pad token
predictions = (1/(1 + np.exp(-predictions))>0.5)*all_masks


# The code bellow will generate a dataframe of predictions and groundtruths. 
# It is tried to keep the names separated with a comma if there are two or more names in a sentence


all_sen = []
all_name = []
all_name_leb = []

for i in range(len(predictions)):
    sen = all_sentences[i]
    name_tok_idx = np.where(predictions[i]==1)[0]
    name_tok0 = all_inputs[i][name_tok_idx]
    if len(name_tok_idx)!=0:
        name_tok = [name_tok0[0]]
        for k in range(len(name_tok_idx)-1):
            if name_tok_idx[k+1]-name_tok_idx[k]!=1:
                name_tok.extend([16])
            else:
                name_tok.extend([name_tok0[k+1]])
    else:
        name_tok = []
    name = tokenizer.decode(name_tok,skip_special_tokens=True)
    
    name_leb_tok_idx = np.where(all_labels[i]==1)[0]
    name_leb_tok0 = all_inputs[i][name_leb_tok_idx]
    if len(name_leb_tok_idx)!=0:
        name_leb_tok = [name_leb_tok0[0]]
        for k in range(len(name_leb_tok_idx)-1):
            if name_leb_tok_idx[k+1]-name_leb_tok_idx[k]!=1:
                name_leb_tok.extend([16])
            else:
                name_leb_tok.extend([name_leb_tok0[k+1]])
    else:
        name_leb_tok = []
        
    name_leb = tokenizer.decode(name_leb_tok,skip_special_tokens=True)
    
    all_sen.append(sen)
    all_name.append(name)
    all_name_leb.append(name_leb)


# making a dataframe and saving it to a csv file
df = pd.DataFrame({"sentence":all_sen,"pred_name":all_name,"gt_name":all_name_leb})
df.to_csv("./predictions/prediction.csv",index = False)