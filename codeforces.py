from bs4 import BeautifulSoup
from markdown import markdown
import re
import latex2mathml.converter

def latex_to_text(latex_string):
    """ Converts a LaTeX string to plaintext using MathML """
    try:
        mathml = latex2mathml.converter.convert(latex_string)
        soup = BeautifulSoup(mathml, 'html.parser')
        return soup.get_text()
    except Exception as e:
        return latex_string
    
def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext """
    html = markdown(markdown_string)
    soup = BeautifulSoup(html, "html.parser")

    # Handle LaTeX expressions
    for script in soup.find_all('script', {'type': 'math/tex'}):
        latex = script.string
        script.replace_with(latex_to_text(latex))

    # Handle inline LaTeX math expressions
    text = soup.get_text('\n', strip=True)
    text = re.sub(r'\$\$(.*?)\$\$', lambda m: latex_to_text(m.group(1)), text)  # Handle double-dollar LaTeX
    text = re.sub(r'\$(.*?)\$', lambda m: latex_to_text(m.group(1)), text)      # Handle single-dollar LaTeX

    # Remove code and preformatted text tags
    for tag in soup.find_all(['code', 'pre']):
        tag.unwrap()

    text = soup.get_text('\n', strip=True)
    
    # Cleanup extra spaces and newlines
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)

    # Remove any remaining dollar signs and other markdown symbols
    text = re.sub(r'\\[a-zA-Z]+', '', text)  # Remove remaining LaTeX commands
    text = re.sub(r'\${2,}', '', text)       # Remove extra dollar signs
    text = re.sub(r'\$', '', text)           # Remove single dollar signs

    # Ensure proper formatting for Input, Output, and Examples sections
    text = re.sub(r'(?i)(Input|Output|Examples)', r'\n\1\n', text)

    return text
