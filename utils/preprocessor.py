
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")

def preprocess_function(examples):
    
    dic = {}
    dic["input_ids"] = []
    dic['token_type_ids'] = []
    dic['attention_mask'] = []
    dic["labels"] = []
     
    for i in range(len(examples["sentences"])):
        sen_li = examples["sentences"][i].split(" ")
        tok = tokenizer(sen_li, max_length=20, truncation=True)

        dic_input_ids = [2]
        dic_token_type_ids = [0]
        dic_attention_mask = [1]
        dic_labels = [0]

        input_ids =tok["input_ids"]
        token_type_ids =tok['token_type_ids']
        attention_mask=tok['attention_mask']
        label=examples['label'][i]

        for t in range(len(sen_li)):
            input_id_len = len(input_ids[t])
            dic_input_ids.extend(input_ids[t][1:input_id_len-1])
            dic_token_type_ids.extend(token_type_ids[t][1:input_id_len-1])
            dic_attention_mask.extend(attention_mask[t][1:input_id_len-1])
            dic_labels.extend([label[t]]*(input_id_len-2))
    
        
        dic["input_ids"].append(dic_input_ids+[3]+[0]*(256-len(dic_input_ids)-1))
        dic['token_type_ids'].append(dic_token_type_ids+[0]*(256-len(dic_token_type_ids)))
        dic['attention_mask'].append(dic_attention_mask+[1]+[0]*(256-len(dic_attention_mask)-1))
        dic['labels'].append(dic_labels+[0]*(256-len(dic_labels)))
    
    return dic