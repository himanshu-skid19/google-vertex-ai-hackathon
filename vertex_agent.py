import requests
import json
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from  config import project_id,region_id,agent_id
def write_request_data(text, language_code="en", time_zone="America/Los_Angeles"):
    """
    Prepares and writes the request data to a JSON file.

    :param text: Query text for Dialogflow.
    :param language_code: Language of the query.
    :param time_zone: Time zone of the query.
    """
    data = {
        "queryInput": {
            "text": {
                "text": text
            },
            "languageCode": language_code
        },
        "queryParams": {
            "timeZone": time_zone
        }
    }
    with open('request.json', 'w') as file:
        json.dump(data, file)

async def call_dialogflow(session_id):
    
    # put this for random gen.
    session_id = session_id   
    # session_id  ="testfinal16"
    subdomain_region = "us"

    url = f"https://{region_id}-dialogflow.googleapis.com/v3/projects/{project_id}/locations/{region_id}/agents/{agent_id}/sessions/{session_id}:detectIntent"

    # Load your request data from a JSON file
    with open('request.json', 'r') as file:
        data = json.load(file)

    # Authenticate and create an access token
    credentials = service_account.Credentials.from_service_account_file(
        'credentials.json'  # Update the path to your service account key file
    )
    scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])
    auth_req = Request()
    scoped_credentials.refresh(auth_req)
    headers = {
        'Authorization': f'Bearer {scoped_credentials.token}',
        'x-goog-user-project': project_id,
        'Content-Type': 'application/json; charset=utf-8'
    }

    # Make the POST request
    response = requests.post(url, headers=headers, json=data)
    return response.json()


