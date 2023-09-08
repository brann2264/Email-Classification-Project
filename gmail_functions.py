import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64


SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
creds = Credentials.from_authorized_user_file('token.json', SCOPES)
service = build('gmail', 'v1', credentials=creds)


def list_messages(service):

    try:
        return service.users().messages().list(userId="me", q="has:nouserlabels").execute()
    except Exception as error:
        print(f"{error}")

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


def main():

    service = build("gmail", "v1", credentials=creds)
    results = service.users().messages().list(userId="me", q="has:nouserlabels").execute()
    messages = results.get("messages", [])

    for message in messages:
        break

    print(get_message(service=service, id=messages[0]["id"]))

main()