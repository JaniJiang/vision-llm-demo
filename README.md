A demonstration of an end-to-end **multimodal AI pipeline**.  
This project integrates **image + text inputs** with a large language model (LLM) to perform vision-language reasoning.  

---

## 🚀 Features
- Convert images to **Base64** for model compatibility
- Combine **image + natural language queries** into a single prompt
- Call an **OpenAI-compatible multimodal API** for inference
- Output **structured model responses**
- Includes workflow diagram for clarity

---
## 🖼 multimodal_pipeline
<img width="502" height="457" alt="Screenshot 2025-09-26 at 6 55 41 PM" src="https://github.com/user-attachments/assets/aef25fee-961a-4cb9-844e-e9469d394f64" />

## 📂 Project Structure

## 📖 Example Code

Below is the core pipeline demonstrating how the system converts an image to Base64,  
builds a multimodal prompt, sends it to the LLM API, and prints the model response.

```python
import base64
from openai import OpenAI
import json

openai_api_key = "EMPTY"
openai_api_base = "https://lpai-inference-guan.inner.chj.cloud/inference/ss-sai/wzr-llm-e650-spdqmc/v1"
model_name = "wzr-llm"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def file_to_base64(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')

def predict_image(image, text):
    base64_image = file_to_base64(image)
    base64_image = f"data:image/png;base64,{base64_image}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image},
                    "max_pixels": 2368 * 1948,
                },
                {"type": "text", "text": text}
            ],
        }
    ]

    chat_response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=messages,
        max_tokens=1024,
        top_p=1,
    )

    raw_res = chat_response.choices[0].message.content
    return raw_res

image_1 = '/path/to/your/image.png'
query = "What is in the image?"

print(predict_image(image_1, query))

Installation
git clone https://github.com/YOUR_USERNAME/vision-llm-demo.git
cd vision-llm-demo

pip install -r requirements.txt

▶️ Usage

Place your test image inside the project folder.

Run the script:

python main.py


Example output:

🔍 Model Output:
The image shows a chart with increasing sales from 2020 to 2023.

📌 Future Work

Support multiple images in a single query

Add RAG (retrieval-augmented generation) for knowledge grounding

Build a simple web UI (Flask/FastAPI)

Batch process evaluation datasets
