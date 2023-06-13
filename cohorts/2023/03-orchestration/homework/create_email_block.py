import os
import json
from prefect_email.credentials import EmailServerCredentials

def create_email_creds_block():
    json_file_path = os.getenv("EMAIL_CREDENTIALS_JSON_FILE_PATH")
    with open(json_file_path) as f:
        email_info = json.load(f)

    credentials = EmailServerCredentials(
        username=email_info["email"],
        password=email_info["password"],  # must be an app password
    )
    credentials.save(name="email-server-credentials", overwrite=True)

if __name__ == "__main__":
    create_email_creds_block()