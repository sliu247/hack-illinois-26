
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
        model_name = "Qwen/Qwen2.5-7B-Instruct"
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
            messages = [
                {"role": "system", "content": (
                    "You are a gossip blogger explaining academic papers "
                    "to someone with zero technical background. Rules:\n"
                    "1) Rewrite ALL technical jargon with everyday analogies that a 5th grader would understand.\n"
                    "2) Keep core ideas but explain using modern slang, dramatic reactions, and short punchy sentences.\n"
                    "3) Never use the original technical terms without explaining them simply first.\n"
                    "4) NEVER use swear words or profanity.\n"
                    "5) Be sensitive to serious or triggering topics (e.g. violence, trauma, tragedy, discrimination). "
                    "Do NOT use dramatic or gossipy language that devalues or makes light of these heavy subjects. Maintain a respectful tone for sensitive content.\n"
                    "6) NEVER include a sign-off, outro, or concluding phrase (like 'XOXO', 'Catch you later', etc.). Do not conclude the post, just stop abruptly when the information ends.\n"
                    "Do NOT get more formal or serious unless the topic requires it. Stay gossipy and fun the rest of the time."
                )},
                {"role": "user", "content": f"Rewrite this text as a gossip text to your best friend:\n\n{chunk}"}
            ]
        else:
            messages = [
                {"role": "system", "content": (
                    "You are a gossip blogger explaining academic papers. "
                    "CRITICAL RULES:\n"
                    "1) Start your response as a direct continuation of the previous paragraph. "
                    "Do NOT start with a greeting, introduction, or summary of what came before.\n"
                    "2) Do NOT repeat ANY information already covered.\n"
                    "3) MATCH THE EXACT SAME ENERGY and tone as the text above — "
                    "same level of modern slang and dramatic reactions and casual vibes.\n"
                    "4) Keep the core ideas but explain them like you're talking to a 5th grader.\n"
                    "5) Never use the original technical terms without explaining them simply first.\n"
                    "6) NEVER use swear words or profanity.\n"
                    "7) Be sensitive to serious or triggering topics (e.g. violence, trauma, tragedy, discrimination). "
                    "Do NOT use dramatic or gossipy language that devalues or makes light of these heavy subjects.\n"
                    "8) NEVER include a sign-off, outro, or concluding phrase (like 'XOXO', 'Catch you later', etc.). Do not conclude the post, just stop abruptly when the information ends.\n"
                    "Do NOT get more formal or serious unless the topic requires it. Stay gossipy and fun the rest of the time."
                )},
                {"role": "user", "content": (
                    f"Here is what you just wrote in your gossip blog post:\n\n"
                    f"\"{prev_gossip[-500:]}\"\n\n"
                    f"Now continue the blog post for this next section without repeating any information already covered or reintroducing yourself."
                    f"Just keep going, explaining jargon with simple analogies.\n\n"
                    f"{chunk}"
                )}
            ]

        # Use the model's official chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(**inputs, max_new_tokens=500, temperature=1.3, do_sample=True)

        # Decode only the generated tokens (everything after the prompt)
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
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
