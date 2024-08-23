import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from datasets import Dataset
import os  

#load .env file
from dotenv import load_dotenv
load_dotenv(override=True)

# Set up the credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("./.local/ep-chatbot-test-data-a79c3ef1fe87.json", scope)
client = gspread.authorize(creds)

# Open the Google Sheet
print(client.list_spreadsheet_files())
sheet = client.open("RAG_Test_Dataset").sheet1


# Get all records from the sheet
data = sheet.get_all_records()

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Convert the pandas DataFrame to a Hugging Face dataset
hf_dataset = Dataset.from_pandas(df)

#print(df.head())  # Check the first few rows of the DataFrame
# Check the dataset
print(hf_dataset)

# Or push to the Hugging Face Hub
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

hf_dataset.push_to_hub("rlarosa/EP-Chatbot-Test-Data", token=HUGGINGFACE_TOKEN)

print("Upload to Hugging Face Hub successful!")