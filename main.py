
# from typing import List
import modal
import sys

app = modal.App("gossip-ai")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "pdfplumber", "accelerate", "PyMuPDF")
    .add_local_file("/Users/sophieliu/Desktop/CS projects/hack-illinois-26/pdf_utils.py", remote_path="/root/pdf_utils.py")
)

@app.function(
    image=image, 
    gpu="A10G",
    timeout=300, 
    scaledown_window=300
)
def chunks_to_gossip(pdf_bytes: bytes):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    global model, tokenizer

    # Load model once per container (saves credits)
    if "model" not in globals():
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto")

    # Read PDF on the GPU container
    from pdf_utils import process_pdf_bytes
    text_chunks = process_pdf_bytes(pdf_bytes)

    gossips = []
    prev_gossip = ""
    for i, chunk in enumerate(text_chunks):
        # edit prompt

        if i == 0:
            prompt = (
                f"[INST] You are a gossip blogger explaining academic papers "
                f"to someone with zero technical background. Rewrite the following "
                f"text as if you're texting your best friend. Rules: "
                f"1) Replace ALL technical jargon with simple everyday analogies "
                f"(e.g. 'Bayesian inference' becomes 'basically a smart guessing game'). "
                f"2) Use modern slang, dramatic reactions, and short punchy sentences. "
                f"3) Keep the core ideas but explain them like you're talking to a 5th grader. "
                f"4) Never use the original technical terms without explaining them simply first.\n\n"
                f"{chunk} [/INST]"
            )
        else:
            prompt = (
                f"[INST] Here is what you just wrote in a gossip blog post:\n\n"
                f"\"{prev_gossip[-500:]}\"\n\n"
                f"Continue the blog post for this next section. "
                f"Keep the SAME gossipy, fun, dramatic tone. "
                f"Do NOT repeat anything or re-introduce yourself. "
                f"Just keep going, explain jargon with simple analogies.\n\n"
                f"{chunk} [/INST]"
            )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens=500, temperature=1.3, do_sample=True)

        result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        gossips.append(result)
        prev_gossip = result

    return gossips

@app.local_entrypoint()
def main():
    # Get PDF file path from command line
    pdf_path = '/Users/sophieliu/Desktop/HackIllinois 2026 Ideas.pdf'

    # Read file directly into bytes locally, 
    # to avoid needing PyMuPDF on the local laptop entirely.
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Send whole PDF bytes to Modal GPU
    gossip_list = chunks_to_gossip.remote(pdf_bytes)

    with open("gossip_results.txt", "w", encoding="utf-8") as out_file:
        for g in gossip_list:
            out_file.write(g + "\n\n")
            print(g)

if __name__ == "__main__":
    app.run(main) 
# '/Users/sophieliu/Desktop/research/research papers/A Markov chain Monte Carlo-based Bayesian framework for system identification and uncertainty estimation of full-scale structures.pdf'
