from dotenv import load_dotenv
from os import environ

load_dotenv()

print(environ['laads_alldata_url'])

load_dotenv('.env.secrets')

print(environ['my_laads_token'])