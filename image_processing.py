import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import PIL.Image
from config import GOOGLE_API_KEY
# def to_markdown(text):
#   text = text.replace('•', '  *')
#   return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

async def recognize_image(img_path,message):
    img = PIL.Image.open(img_path)
    response = model.generate_content([message, img], stream=True)
    response.resolve()
    return response.text