import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import BertModel
from contextlib import nullcontext

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bertbase = BertModel.from_pretrained("bert-base-cased")
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 2)
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        output_1 = self.bertbase(input_ids, token_type_ids, attention_mask)
        output_2 = self.l2(output_1[1])
        output = self.l3(output_2)
        return output


def subtext_data(data, length=512):
    
    segmented_inputs = []
    return_list = []
    length = length-1
    
    #Removes [CLS] and [SEP] key
    data = data["input_ids"][1:-1]
    data_length = len(data)
    
    for num in np.arange(int(data_length/length)+1):
        
        #if last subtext
        if (num+1)*length > data_length:
            padded_segment = np.concatenate((data[length*num:],np.zeros((num+1)*length-data_length)))
            padded_cls_comment = np.concatenate(([101], padded_segment))
            segmented_inputs.append(padded_cls_comment)
            
        else:
            data_cls_comment = np.concatenate(([101],data[length*num:length*(num+1)]))
            segmented_inputs.append(data_cls_comment)
    
    for subtext in segmented_inputs:
        return_dict = {"input_ids": torch.tensor(subtext, dtype=torch.int64), 
                       "token_type_ids": torch.zeros(length+1, dtype=torch.int64), 
                       "attention_mask": torch.ones(length+1, dtype=torch.int64)}
        return_list.append(return_dict)
        
    return return_list


class phising_text_dataset(Dataset):
    
    def __init__(self, data, tokenizer, subtexter=None, max_length=512):
        self.data = data
        self.subtexter = subtexter
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = {"Commercial Spam":1, "False Positives ":0, "Fraud": 1, "Phishing":1}
    
    def __len__(self):
        return int(self.data.size/2)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        
        text = self.tokenizer(text)
        
        if self.subtexter:
            text = self.subtexter(text, self.max_length)
            
        return text, label


def nlp_collate_fn(data):
    
    length = len(data[0][0][0]["input_ids"])
    input_ids = torch.empty((0, length), dtype=torch.int64)
    token_type_ids = torch.empty((0, length), dtype=torch.int64)
    attention_mask = torch.empty((0, length), dtype=torch.int64)
    labels = []
    
    for email in data:
        for subtext in email[0]:
            input_ids = torch.vstack((input_ids, subtext["input_ids"]))
            token_type_ids = torch.vstack((token_type_ids, subtext["token_type_ids"]))
            attention_mask = torch.vstack((attention_mask, subtext["attention_mask"]))
            labels += [email[1]]
            
    return_dict = {"input_ids": input_ids,
                   "token_type_ids": token_type_ids,
                   "attention_mask": attention_mask,
                   "labels": torch.tensor(labels)}
    
    return return_dict


def bert_train_step(model, dataloader, loss_fn, optimizer, accuracy_fn, device = "cpu", test=False):
    """
    Performs one epoch of training on the model.

    Parameters
    -----------------------
    model: torch.nn.modules
    dataloader: torch.utils.data.DataLoader instances
    loss_fn: torch.nn
    accuracy_fn: torchmetrics.Accuracy instances
    device: str | "cpu", "cuda", or "mps" 
    test: bool | True for evaluation mode and no grad
    -----------------------
    Returns
    epoch_loss, epoch_acc: float | averaged loss and accuracy across the epoch
    """

    model = model.to(device)
    accuracy_fn = accuracy_fn.to(device)

    epoch_loss = 0
    epoch_acc = 0
    total_subtexts = 0

    if test:
        model.eval()
    else:
        model.train()

    with torch.inference_mode() if test else nullcontext():
            
        for batch, data_dict in enumerate(dataloader):

            input_ids = data_dict["input_ids"].to(device)
            token_type_ids = data_dict["token_type_ids"].to(device)
            attention_mask = data_dict["attention_mask"].to(device)
            labels = data_dict["labels"].to(device)
         
            pred_prob = model(input_ids, token_type_ids, attention_mask)
            loss = loss_fn(pred_prob, labels)

            if test == False:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            prediction = torch.max(torch.softmax(pred_prob, dim=1), dim=1).indices
            accuracy = accuracy_fn(prediction, labels)
            
            total_subtexts += len(input_ids)
            epoch_loss += loss*len(input_ids)
            epoch_acc += accuracy*len(input_ids)
                
        epoch_loss /= total_subtexts
        epoch_acc /= total_subtexts

    return epoch_loss, epoch_acc  


def model_evaluate(model, tokenizer, text, subtexter):
    
    model.eval()
    model.to("cpu")
    
    results = []
    
    with torch.inference_mode():
        
        tokenized = tokenizer(text)
        subtexted = subtexter(tokenized)

        for subtext_num in range(len(subtexted)):
            inputs = subtexted[subtext_num]["input_ids"].unsqueeze(dim=0)
            token = subtexted[subtext_num]["token_type_ids"].unsqueeze(dim=0)
            mask = subtexted[subtext_num]["attention_mask"].unsqueeze(dim=0)

            out = model(inputs, token, mask)
            prediction = [int(torch.argmax(torch.softmax(out, 1), dim=1))]
            results += prediction

    return results


