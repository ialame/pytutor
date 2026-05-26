import os
from dotenv import load_dotenv; load_dotenv()
from langsmith import Client
c = Client()
print('endpoint:', c.api_url)
print('projet :', os.getenv('LANGSMITH_PROJECT'))
print('whoami :', c.read_shared_dataset.__self__.info if False else 'ok')
print('test trace...')
from langsmith import traceable
@traceable
def ping(): return 'pong'
ping()
print('→ envoyé. Va voir dans l UI sous le projet PyTutor2')