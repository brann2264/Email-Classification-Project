import os.path
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def get_message(service, id):
    """
    Gets the message corresponding to the input ID and returns it

    Parameters:
    -----------
    service: gmail service
    id: int
    -----------
    Returns:
    text: str of the email in plain text
    """
    try: 
        message = service.users().messages().get(userId="me", id=id).execute()

        #Obtains only the plain text portion of the email and decodes it to utf-8
        for part in message["payload"]["parts"]:
            if part["mimeType"] == "text/plain":
                text = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
                return text

    except Exception as error:
        print(f"Message not found: Error - {error}")

def create_label(service, label_name, visibility="show"):
    """
    Creates a label with input name and visibility and returns it

    Parameters:
    -----------
    service: gmail service
    label_name: str, name of the label
    visibility: "show" or "hide", determines the visibility of the labeled messages
    -----------
    Returns:
    label: gmail label instance
    """
    
    label_body = {"name": label_name,
                "messageListVisibility": visibility,
                "labelListVisibility": "labelShow"}  
    
    label = service.users().labels().create(userId="me", body=label_body).execute

    return label

def label_email(service, label, email_id):
    """
    Labels the target message with corresponding input id with the input label and returns it

    Parameters:
    -----------
    service: gmail service
    label: gmail label instance
    email_id: int
    -----------
    Returns:
    labeled_message: gmail message instance
    """

    request_body = {"addLabelIds":[label]}
    labeled_message = service.users().messages().modify(userId="me", id=email_id, body=request_body)

    return labeled_message
