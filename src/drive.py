from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

import os
import pickle

# # Step 1: Authenticate and create the Drive API client
# def authenticate_drive():
#     SCOPES = ['https://www.googleapis.com/auth/drive.file']
#     creds = None
    
#     # Load credentials if they already exist
#     if os.path.exists('token.pickle'):
#         with open('token.pickle', 'rb') as token:
#             creds = pickle.load(token)
    
#     # If no valid credentials, log in and save them
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(
#                 'credentials.json', SCOPES)
#             creds = flow.run_local_server(port=0)
#         with open('token.pickle', 'wb') as token:
#             pickle.dump(creds, token)
    
#     return build('drive', 'v3', credentials=creds)

# # Step 2: Upload a file to Google Drive
# def upload_file(service, file_path, folder_id=None):
#     file_metadata = {'name': os.path.basename(file_path)}
#     if folder_id:
#         file_metadata['parents'] = [folder_id]
    
#     media = MediaFileUpload(file_path, resumable=True)
#     file = service.files().create(
#         body=file_metadata,
#         media_body=media,
#         fields='id'
#     ).execute()
    
#     print(f"File uploaded successfully! File ID: {file['id']}")

# # Main function
# if __name__ == '__main__':
#     # Authenticate and get the Drive service
#     service = authenticate_drive()
    
#     # Path to the file or folder to upload
#     file_path = 'path/to/your/file_or_folder'
    
#     # Upload the file
#     upload_file(service, file_path)
