import modal
<<<<<<< HEAD

app = modal.App("example-get-started")


@app.function()
def square(x):
    print("This code is running on a remote worker!")
    return x**2


@app.local_entrypoint()
def main():
    print("the square is", square.remote(42))
=======
import os, openai
import sys
from pdf_utils import process_pdf

# try:
#     from config import OPENAI_API_KEY
# except ImportError:
#     raise ImportError("Please set your OpenAI API key in config.py")
app = modal.App("gossip-ai")
image = modal.Image.debian_slim().pip_install(["openai"])

@app.function(image=image, timeout=300)
def chunks_to_gossip(text_chunks: list):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    gossips = []
    for chunk in text_chunks:
        # edit prompt

        prompt = f"Turn this academic/historical text into short, gossipy style:\n\n{chunk}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        gossips.append(response.choices[0].message['content'])
    return gossips

# Get PDF file path from command line
pdf_path = sys.argv[1]

# Process PDF into chunks
text_chunks = process_pdf(pdf_path)

# Send chunks to your gossip AI function

handle = chunks_to_gossip.spawn(text_chunks)
gossip_list = handle.get()

# Print results
for g in gossip_list:
    print(g)
# '/Users/sophieliu/Desktop/research/research papers/A Markov chain Monte Carlo-based Bayesian framework for system identification and uncertainty estimation of full-scale structures.pdf'


>>>>>>> aa46100 (update)
