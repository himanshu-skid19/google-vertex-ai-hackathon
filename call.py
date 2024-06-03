import vertexai
import os
from vertexai.preview import reasoning_engines

PID = os.environ.get("projectid")
REID = os.environ.get("reasoningEngineid")

remote_app = vertexai.preview.reasoning_engines.ReasoningEngine('projects/197837456592/locations/us-central1/reasoningEngines/2310944743166574592')
response = remote_app.query(input="What is the exchange rate from Singapore dollars to Indonesian Rupiah currency?")

print(response)
#service-197837456592