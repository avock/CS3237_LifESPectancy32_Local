import os
import csv
from dotenv import load_dotenv
import requests

load_dotenv()

TEST_HEADERS = ['header1', 'header2', 'header3']
TEST_DATA = {
    'header1': 'value1',
    'header2': 'value2',
    'header3': 'value3'
}

bot_token = os.environ.get("BOT_TOKEN")
chat_id = os.environ.get("CHAT_ID_CK")

def write_to_csv(csv_filename, headers=TEST_HEADERS, data=TEST_DATA):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(parent_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_file_path = os.path.join(data_dir, csv_filename)

    if not os.path.isfile(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)

    with open(csv_file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([data.get(header, '') for header in headers])
        
def send_telegram_message(message):
    apiURL = f'https://api.telegram.org/bot{bot_token}/sendMessage'

    try:
        response = requests.post(apiURL, json={'chat_id': chat_id, 'text': message})
        # print(response.text)
    except Exception as e:
        print(e)
        
def process_json_payload(payload_json, keys):
    extracted_data = {}
    for key in keys:
        if key in payload_json:
            extracted_data[key] = payload_json[key]

    return extracted_data