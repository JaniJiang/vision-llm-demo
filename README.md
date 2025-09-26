Multimodal Image Understanding with LLM

This project demonstrates a **multimodal pipeline** that integrates image processing and large language models (LLMs).  
It allows users to **upload an image + ask a textual question**, and the model will generate a response based on both inputs.

                                   multimodal image Understanding Pipeline
<img width="503" height="480" alt="Screenshot 2025-09-26 at 7 09 13 PM" src="https://github.com/user-attachments/assets/b739bc63-3749-41fb-88ef-04c2bdd42031" />

--- 

Features
- Convert images to **Base64** for model compatibility
- Combine **image + natural language queries** into a single prompt
- Call an **OpenAI-compatible multimodal API** for inference
- Output **structured model responses**
- Includes workflow diagram for clarity

---


Project Structure

Example Code

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


---

## ⚙️ Installation
```bash
git clone https://github.com/YOUR_USERNAME/vision-llm-demo.git
cd vision-llm-demo

pip install -r requirements.txt
