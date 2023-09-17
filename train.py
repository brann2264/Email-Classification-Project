from model_functions import BERTClass, subtext_data, phising_text_dataset, nlp_collate_fn, bert_train_step

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy

#Load dataset
datadf = pd.read_csv("/kaggle/input/fraud-email-dataset/fraud_email_.csv")
#Remove incompatible data point
datadf = datadf.drop(5129)

torch.manual_seed(5)

#Prepares data into training and testing dataloaders
train_df, test_df= train_test_split(datadf.sample(500), random_state=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

train_data = phising_text_dataset(data=train_df,
                                  tokenizer=tokenizer,
                                  subtexter=subtext_data,
                                  max_length=512)
test_data = phising_text_dataset(data=test_df,
                                 tokenizer=tokenizer,
                                 subtexter=subtext_data,
                                 max_length=512)

train_dataloader = DataLoader(train_data, batch_size=6, shuffle=True, collate_fn=nlp_collate_fn)
test_dataloader = DataLoader(test_data, batch_size=6, collate_fn=nlp_collate_fn)

#Initializes the model, loss function, optimizer, and accuracy
model = BERTClass()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
accuracy_fn = Accuracy("multiclass", num_classes=2)

epochs = 10

#Trains the model for a number of epochs
for epoch in tqdm(range(epochs)):
    
    loss, acc = bert_train_step(model=model, 
                               dataloader=train_dataloader, 
                               loss_fn=loss_fn, 
                               optimizer=optimizer, 
                               accuracy_fn=accuracy_fn, 
                               device="cuda", 
                               test=False)
    
    if (epoch+1)%3 == 0: 
        val_loss, val_acc = bert_train_step(model=model, 
                                           dataloader=test_dataloader, 
                                           loss_fn=loss_fn, 
                                           optimizer=optimizer, 
                                           accuracy_fn=accuracy_fn, 
                                           device="cuda", 
                                           test=True)
        
        print(f"Epoch:{epoch}|Loss:{loss:.4f}|Accuracy:{acc:.4f}|Validation Loss:{val_loss:.4f}| Validation Acc:{val_acc:.4f}")
    
    else:
        print(f"Epoch:{epoch}|Loss:{loss:.4f}|Accuracy:{acc:.4f}")
    
#save_dir = Path("fraud_email_statedict2")
#torch.save(model.state_dict(), save_dir)


