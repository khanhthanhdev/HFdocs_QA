from llama_index.core.llms import ChatMessage
from llama_index.llms.gemini import Gemini
from dotenv import load_dotenv
import os
from llama_index.core import Settings
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

messages = [
    ChatMessage(role="user", content="Hello friend!"),
    ChatMessage(role="assistant", content="Yarr what is shakin' matey?"),
    ChatMessage(
        role="user", content="Help me decide what to have for dinner."
    ),
]
resp = Gemini().chat(messages)
print(resp)
