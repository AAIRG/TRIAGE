import configparser
from settings import ROOT_DIR
import os

print("Configuring API keys...")
# Parse config.ini file
config = configparser.ConfigParser()
PATH_CONFIG = os.path.join(ROOT_DIR, 'config.ini')
config.read(PATH_CONFIG)

# Extract API keys
openai_api_key = config['OPENAI']['token']
openai_org_id = config['OPENAI']['org_id']
lambda_api_key = config['LAMBDA']['api_token']

class NeptuneConfig:
    PROJECT_1 = config['neptune']['project_1']
    PROJECT_2 = config['neptune']['project_2']
    API_TOKEN = config['neptune']['api_token']
