import os.path
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def get_message(service, id):

    try: 
        message = service.users().messages().get(userId="me", id=id).execute()

        for part in message["payload"]["parts"]:
            if part["mimeType"] == "text/plain":
                text = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                print(text)
                return text

    except Exception as error:
        print(f"Message not found: Error - {error}")

def create_label(service, label_name, visibility="show"):
    
    label_body = {"name": label_name,
                "messageListVisibility": visibility,
                "labelListVisibility": "labelShow"}  
    
    label = service.users().labels().create(userId="me", body=label_body).execute

    return label

def label_email(service, label, email_id):

    request_body = {"addLabelIds":[label]}
    labeled_message = service.users().messages().modify(userId="me", id=email_id, body=request_body)

    return labeled_message
