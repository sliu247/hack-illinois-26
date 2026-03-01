
# from typing import List
import modal
import sys

app = modal.App("gossip-ai")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "pdfplumber", "accelerate", "PyMuPDF", "fastapi", "python-multipart")
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

# ------ WEB APP CODE ------
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse

web_app = FastAPI()

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Academic - Spilling Facts</title>
<link href="https://fonts.googleapis.com/css2?family=Caveat:wght@400;700&family=Kalam:wght@300;400;700&display=swap" rel="stylesheet">
<style>
    * { box-sizing: border-box; }
    body {
        margin: 0; padding: 40px;
        background-color: #ebd5c1;
        color: #3e332a;
        font-family: 'Kalam', cursive;
        min-height: 100vh;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .header-container {
        text-align: center;
        width: 100%;
        max-width: 800px;
        margin-bottom: 30px;
    }
    .title-row {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 15px;
        font-family: 'Caveat', cursive;
        font-size: 5rem;
        font-weight: 700;
        margin-bottom: 0px;
        line-height: 1;
    }
    .coffee-icon {
        font-size: 4rem;
    }
    .subtitle {
        font-family: 'Caveat', cursive;
        font-size: 1.5rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 20px;
    }
    .divider {
        border-top: 2px solid #3e332a;
        margin: 10px 0;
    }
    .welcome-text {
        font-size: 2.5rem;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin: 10px 0;
    }
    .main-content {
        display: flex;
        gap: 40px;
        width: 100%;
        max-width: 900px;
        align-items: flex-start;
        margin-top: 30px;
    }
    @media (max-width: 768px) {
        .main-content {
            flex-direction: column;
            align-items: center;
        }
    }
    .upload-box {
        flex: 1;
        border: 2px solid #3e332a;
        border-radius: 4px;
        background-color: transparent;
        min-height: 350px;
        width: 100%;
        max-width: 300px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        cursor: pointer;
        transition: background-color 0.2s;
        position: relative;
    }
    .upload-box:hover {
        background-color: rgba(62, 51, 42, 0.05);
    }
    .upload-box.dragging {
        background-color: rgba(62, 51, 42, 0.1);
        border-style: dashed;
    }
    .upload-text {
        font-size: 2.5rem;
        font-family: 'Caveat', cursive;
        transform: rotate(-10deg);
        text-align: center;
    }
    .file-input { display: none; }
    
    .right-column {
        flex: 2;
        width: 100%;
        display: flex;
        flex-direction: column;
    }
    
    .text-lines {
        width: 100%;
        min-height: 350px;
        background-image: repeating-linear-gradient(transparent, transparent 39px, #3e332a 39px, #3e332a 40px);
        line-height: 40px;
        font-size: 1.3rem;
        padding-top: 5px; /* align text to line */
        white-space: pre-wrap;
        display: none; /* Hidden until output is ready */
    }
    
    .loader-container {
        display: none;
        width: 100%;
        text-align: center;
        margin-top: 20px;
        font-size: 1.5rem;
    }
</style>
</head>
<body>
    <div class="header-container">
        <div class="title-row">
            Academic <span class="coffee-icon">☕</span>
        </div>
        <div class="subtitle">SIPPING TEA & SPILLING FACTS</div>
        
        <div class="divider"></div>
        <div class="welcome-text">WELCOME!</div>
        <div class="divider"></div>
    </div>
    
    <div class="main-content">
        <label for="file-upload" class="upload-box" id="drop-zone">
            <div class="upload-text">upload<br>pdf</div>
            <input id="file-upload" type="file" accept="application/pdf" class="file-input" />
        </label>
        
        <div class="right-column">
            <div id="loader" class="loader-container">
                Brewing the tea... 🍵
            </div>
            <div class="text-lines" id="result-container">
            </div>
        </div>
    </div>

    <script>
        const fileUpload = document.getElementById('file-upload');
        const dropZone = document.getElementById('drop-zone');
        const loader = document.getElementById('loader');
        const resultContainer = document.getElementById('result-container');

        // Handle drag and drop
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });
        function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.add('dragging'), false);
        });
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragging'), false);
        });
        dropZone.addEventListener('drop', (e) => {
            let dt = e.dataTransfer;
            let files = dt.files;
            handleFile(files[0]);
        });
        
        fileUpload.addEventListener('change', function(e) {
            handleFile(e.target.files[0]);
        });

        async function handleFile(file) {
            if (!file) return;

            dropZone.style.pointerEvents = 'none';
            resultContainer.innerHTML = '';
            resultContainer.style.display = 'none';
            loader.style.display = 'block';

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/upload', { method: 'POST', body: formData });
                const data = await response.json();
                resultContainer.textContent = data.result;
                resultContainer.style.display = 'block';
            } catch (error) {
                resultContainer.textContent = 'Oops! Something went wrong.';
                resultContainer.style.display = 'block';
            } finally {
                loader.style.display = 'none';
                dropZone.style.pointerEvents = 'auto';
            }
        }
    </script>
</body>
</html>
"""

@web_app.get("/")
def index():
    return HTMLResponse(content=html_content)

@web_app.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    res = chunks_to_gossip.remote(contents)
    return {"result": "\n\n".join(res)}

web_image = modal.Image.debian_slim().pip_install("fastapi", "python-multipart", "uvicorn")

@app.function(image=web_image)
@modal.asgi_app()
def fastapi_app():
    return web_app

@app.local_entrypoint()
def main():
    print("Run `modal serve main.py` and click the web endpoint URL to view the UI!")
