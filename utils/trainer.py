import torch
import torch.nn as nn
import numpy as np  
from transformers import Trainer

'''
    In our case, we are doning Bengali NER only for Persons. 
    As it is a binary classinfication we will have to change 
    the loss calculation process in the trainer. Also, token's 
    are imbalanced here. So, we will use a F1 score a our metric. 
    For this purpose we will write another function here.
'''

class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.my_loss = nn.BCELoss(reduction = 'none')
        self.sigmoid = nn.Sigmoid()
        
    def compute_loss(self, model, inputs,return_outputs=False):

        outputs = model(**inputs)
        outputs.logits = outputs.logits[:,:,0]
        att_mask = inputs.attention_mask.type(torch.float32) # attention mask will be used to exclude redundent portion from loss calculation
        
        loss = torch.sum(self.my_loss(self.sigmoid(outputs.logits),inputs.labels.type(torch.float32))*att_mask)/torch.sum(att_mask)
        
        return (loss, outputs) if return_outputs else loss
    

def compute_metrics(eval_pred):
    predictions, label_ids, inputs = eval_pred
    pred_mask = (inputs!=2)*(inputs!=3)*(inputs!=0)*1 # This mask will be used to exclude redundent parts from calculation
    predictions = (1/(1 + np.exp(-predictions))>0.5)*1

    TP = np.sum((label_ids==predictions)*label_ids*pred_mask)
    FP = np.sum((label_ids!=predictions)*predictions*pred_mask)
    FN = np.sum((label_ids!=predictions)*label_ids*pred_mask)
    
    F1 = TP/(TP+0.5*(FP+FN)+1e-10)
    
    result = {}
    result["F1"]=F1
    
    return {k: round(v, 4) for k, v in result.items()}