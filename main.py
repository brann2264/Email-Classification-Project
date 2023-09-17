from __future__ import print_function
from model_functions import BERTClass, subtext_data, model_evaluate

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from gmail_functions import get_message, create_label, label_email
from transformers import AutoTokenizer
import torch

def main():

    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly',
             "https://www.googleapis.com/auth/gmail.modify"]
        
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    service = build("gmail", "v1", credentials=creds)

    #Obtains list of messages, but not their text body
    results = service.users().messages().list(userId="me", q="has:nouserlabels").execute()
    messages = results.get("messages", [])

    #Create "safe" and "malicious" labels
    safe_label = create_label(service=service, label_name="Safe")
    malicious_label = create_label(service=service, label_name="Malicious", visibility="hide")

    #Initializes the tokenizer and trained model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = BERTClass()
    model.load_state_dict(torch.load("fraud_email_statedict2"))

    #For each message found
    for message in messages:
        #Obtains the text of the message
        message_id = message["id"]
        message_text = get_message(service=service, id=message_id)
        
        #Evaluates the message using the model
        prediction = model_evaluate(model=model, tokenizer=tokenizer, text=message_text, subtexter=subtext_data)
        
        #Prediction is based on the class (safe/malicious) with the higher amount of predictions among the subtexts
        label = round(sum(prediction))

        #Labels the email based on the prediction {0: "Malicious", 1: "Safe"}
        if label:
            labeled_email = label_email(service=service, label=safe_label, email_id=message_id)
        else:
            labeled_email = label_email(service=service, label=malicious_label, email_id=message_id)

main()