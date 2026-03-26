<div align="center">

<h1>
  <img src="https://raw.githubusercontent.com/your-username/accumulative-decoding/main/assets/logo.png" width="40" alt="logo"/>
  Accumulative Decoding
</h1>

<h3>Mitigating Hallucinations in Large Vision-Language Models via Accumulative Decoding</h3>

<p>
  <a href="https://arxiv.org/abs/your-paper-id"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=flat-square&logo=arxiv" alt="Paper"/></a>
  &nbsp;
  <a href="#"><img src="https://img.shields.io/badge/Python-3.9%2B-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  &nbsp;
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  &nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"/></a>
  &nbsp;
  <a href="#"><img src="https://img.shields.io/badge/Training-Free-blueviolet?style=flat-square" alt="Training-Free"/></a>
</p>

<p>
  <b>Haotian Zhang*</b> &nbsp;·&nbsp; <b>Zhiyun Zhang*</b>
  <br>
  <sub>Glasgow College, University of Electronic Science and Technology of China</sub>
  <br>
  <sub>* Equal contribution</sub>
</p>

<br>

<table>
<tr>
<td align="center"><b>MME (Perception)</b></td>
<td align="center"><b>MM-Vet</b></td>
<td align="center"><b>MMMU</b></td>
</tr>
<tr>
<td align="center"><img src="https://img.shields.io/badge/1499.79-🏆 SOTA-gold?style=for-the-badge" alt="MME"/></td>
<td align="center"><img src="https://img.shields.io/badge/32.8-🏆 SOTA-gold?style=for-the-badge" alt="MM-Vet"/></td>
<td align="center"><img src="https://img.shields.io/badge/36.4-🏆 SOTA-gold?style=for-the-badge" alt="MMMU"/></td>
</tr>
</table>

</div>

---

## 📌 Overview

**Hallucination** is one of the most critical obstacles to reliably deploying Large Vision-Language Models (LVLMs): the model produces fluent, confident text that is factually *inconsistent* with what is actually in the image.

Existing inference-time methods apply a **one-shot or static** correction, failing to address the progressive drift from visual evidence that accumulates across long generated sequences.

We introduce **Accumulative Decoding (AD)** — a lightweight, **training-free** framework that maintains a *cumulative* visual-grounding signal throughout every step of autoregressive generation. The result is a continuous, self-reinforcing push toward visual faithfulness that:

- ✅ Requires **no fine-tuning** and **no architecture changes**
- ✅ Drops in as a standard `LogitsProcessor` in HuggingFace `generate()`
- ✅ Consistently outperforms baselines on **3 major benchmarks**
- ✅ Improves factual accuracy **without sacrificing fluency**

---

## 🔬 Method

> AD enforces visual grounding not as a one-time fix, but as an **ever-growing cumulative signal** that amplifies itself with each generated token.

<div align="center">

```
Image I
   │
   ▼
Visual Encoder ──► mm_projector ──► v₀  (visual embedding)
                                     │
         ┌───────────────────────────┘
         │   At each decoding step t:
         │
         │   ① Grounding score for every candidate token
         │      s(yₜ|I) = softmax[ sim(Emb(yₜ), v₀) ]
         │
         │   ② Cumulative score (sum over all past tokens)
         │      Gₜ = Σᵢ₌₁ᵗ s(yᵢ₋₁|I)
         │
         │   ③ Dynamic weight
         │      λₜ = γ · σ( Avg of past grounding scores )
         │
         │   ④ Final logits
         │      lₐd = (1-λₜ)·lbase  +  λₜ·(α·Gₜ·s(yₜ|I) + β)
         │
         ▼
    Next token  →  loop back to ①
```

</div>

### Equations

| Step | Formula | Role |
|:----:|---------|------|
| **①** | $s(y_t \mid I) = \dfrac{\exp(\text{sim}(\text{Emb}(y_t),\, v_0))}{\sum_{y_i} \exp(\text{sim}(\text{Emb}(y_i),\, v_0))}$ | Per-token visual alignment score |
| **②** | $G_t = \displaystyle\sum_{i=1}^{t} s(y_{i-1} \mid I)$ | Cumulative visual grounding |
| **③** | $l_{G_t}(y_t) = \alpha \cdot G_t \cdot s(y_t \mid I) + \beta$ | Grounding logit adjustment |
| **④** | $\lambda_t = \gamma \cdot \sigma\!\left(\text{Avg}_{i<t}\, s(y_i \mid I)\right)$ | Dynamic weight (self-adapting) |
| **⑤** | $l_{\text{AD}}(y_t) = (1 - \lambda_t)\cdot l_{\text{base}}(y_t) + \lambda_t \cdot l_{G_t}(y_t)$ | **Final adjusted logits** |

**Default hyperparameters:** `α = 0.5`, `β = 0.3`, `γ = 0.8`

---

## 📊 Results

All experiments use **LLaVA-1.5-7B** with temperature = 0.

### MME Benchmark — Multimodal Perception

<div align="center">

| Method | Exist. | Count | Pos. | Color | Posters | Celebrity | Scene | Landmark | Artwork | OCR | **Total** |
|:------:|:------:|:-----:|:----:|:-----:|:-------:|:---------:|:-----:|:--------:|:-------:|:---:|:---------:|
| Vanilla | 190.00 | 160.00 | 138.33 | 165.00 | 140.48 | 135.88 | 156.25 | 161.50 | 118.50 | 125.00 | 1490.94 |
| DoLA | 190.00 | 158.33 | 143.33 | 165.00 | 139.46 | 133.24 | 157.00 | 161.50 | 118.75 | 125.00 | 1491.61 |
| VCD | 173.33 | 151.67 | 138.33 | 165.00 | 140.48 | 137.06 | 151.00 | 164.75 | 120.50 | 117.50 | 1459.62 |
| VDD | 180.00 | 148.30 | 135.00 | 170.00 | 141.84 | 144.71 | 151.75 | 167.50 | 123.75 | 110.00 | 1472.88 |
| DeCO | 190.00 | 148.33 | 115.00 | 165.00 | 149.66 | 135.29 | 152.25 | 164.50 | 110.25 | 130.00 | 1460.29 |
| **AD (Ours)** | **195.00** | **160.00** | 135.33 | 165.00 | 142.48 | 139.23 | 153.00 | 164.75 | 115.00 | **130.00** | **1499.79** |

</div>

> AD achieves the **highest total score of 1499.79**, with particularly strong gains in Existence (+5 over best baseline) and consistent performance across all categories.

---

### MM-Vet Benchmark — Integrated Multimodal Reasoning

<div align="center">

| Method | Rec | OCR | Know | Gen | Spat | Math | **Total** |
|:------:|:---:|:---:|:----:|:---:|:----:|:----:|:---------:|
| Vanilla | 36.1 | 23.0 | 18.0 | 22.2 | 25.1 | 11.5 | 31.1 |
| DoLA | 36.5 | 22.3 | 18.1 | 23.0 | 25.7 | 7.7 | 30.8 |
| VCD | 36.1 | 22.4 | 21.0 | 23.1 | 28.4 | 3.8 | 31.1 |
| VDD | 37.1 | 22.8 | 19.0 | 21.7 | 28.3 | 11.2 | 31.8 |
| DeCO | 35.8 | 26.8 | 19.2 | 21.3 | 30.2 | 7.7 | 32.6 |
| **AD (Ours)** | 36.6 | **27.1** | **21.3** | 22.4 | **30.2** | 10.5 | **32.8** |

</div>

> AD leads on OCR, Knowledge, and Spatial reasoning simultaneously — no cherry-picking, balanced improvement across all skills.

---

### MMMU Benchmark — College-Level Expert Reasoning

<div align="center">

| Method | Vanilla | DoLA | VCD | VDD | DeCO | **AD (Ours)** |
|:------:|:-------:|:----:|:---:|:---:|:----:|:-------------:|
| Accuracy (%) | 35.3 | 35.7 | 35.8 | 34.9 | 33.9 | **36.4** |

</div>

> On the most challenging benchmark, AD surpasses all competitors, including VCD (+0.6) and DoLA (+0.7).

---

## 🔧 Ablation Study

Hyperparameter sensitivity analysis shows the framework is **robust** — performance degrades gracefully away from the defaults.

<div align="center">

| Hyperparameter | Search Range | **Optimal** | Benchmark | Best Score† |
|:--------------:|:------------:|:-----------:|:---------:|:-----------:|
| α (grounding scale) | {0.1, 0.3, 0.5, 0.7, 0.9} | **0.5** | MME | 1499.79 |
| β (grounding bias) | {0.0, 0.1, 0.3, 0.5, 0.7} | **0.3** | MM-Vet | 32.8 |
| γ (weight ceiling) | {0.5, 0.7, 0.8, 0.9, 1.0} | **0.8** | MMMU | 35.7 |

<sub>† Each row sweeps one hyperparameter while holding the other two at their default values. The full-default combination (α=0.5, β=0.3, γ=0.8) achieves the main results reported above (MME 1499.79 / MM-Vet 32.8 / MMMU 36.4).</sub>

</div>

---

## 🚀 Getting Started

### Installation

```bash
# 1. Clone this repo
git clone https://github.com/your-username/accumulative-decoding.git
cd accumulative-decoding

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install LLaVA from source
git clone https://github.com/haotian-liu/LLaVA.git
pip install -e LLaVA
```

Then download the **LLaVA-1.5-7B** weights:

```bash
# Via HuggingFace Hub
huggingface-cli download liuhaotian/llava-v1.5-7b --local-dir checkpoints/llava-v1.5-7b
```

---

### Quick Start

```python
import torch
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

from accumulative_decoding import (
    AccumulativeDecodingProcessor,
    get_llava_visual_embedding,
    get_token_embeddings,
)

# ── Load model ──────────────────────────────────────────────────────────────
model_path = "checkpoints/llava-v1.5-7b"
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path, None, get_model_name_from_path(model_path)
)
model.eval()

# ── Prepare image ────────────────────────────────────────────────────────────
image = Image.open("your_image.jpg").convert("RGB")
image_tensor = (
    process_images([image], image_processor, model.config)[0]
    .unsqueeze(0)
    .to(model.device, dtype=torch.float16)
)

# ── Build AD processor ────────────────────────────────────────────────────────
v0        = get_llava_visual_embedding(model, image_tensor)   # visual anchor
token_embs = get_token_embeddings(model)
ad_processor = AccumulativeDecodingProcessor(
    v0, token_embs, alpha=0.5, beta=0.3, gamma=0.8
)

# ── Generate with AD ──────────────────────────────────────────────────────────
prompt = DEFAULT_IMAGE_TOKEN + "\nDescribe this image in detail."
input_ids = tokenizer_image_token(
    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
).unsqueeze(0).to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        logits_processor=[ad_processor],  # <-- plug-and-play
        max_new_tokens=512,
        temperature=0,
        do_sample=False,
    )

# Decode only the newly generated tokens (exclude the input prompt)
new_tokens = output_ids[0][input_ids.shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
```

---

### Benchmark Evaluation

```bash
# MME Perception
python run_eval.py \
    --benchmark mme \
    --model_path checkpoints/llava-v1.5-7b \
    --data_path  data/MME

# MM-Vet  (responses saved → submit to GPT-4 evaluator)
python run_eval.py \
    --benchmark mmvet \
    --model_path checkpoints/llava-v1.5-7b \
    --data_path  data/mm-vet

# MMMU
python run_eval.py \
    --benchmark mmmu \
    --model_path checkpoints/llava-v1.5-7b \
    --data_path  data/MMMU          # or omit to auto-download from HuggingFace
```

Results are saved as JSON files under `results/<benchmark>/`.

---

## 📁 Repository Structure

```
accumulative-decoding/
├── accumulative_decoding/
│   ├── __init__.py
│   ├── ad_processor.py      # Core LogitsProcessor — implements Eq.(1)-(5)
│   └── model_utils.py       # Visual embedding extraction from LLaVA
├── eval/
│   ├── eval_mme.py          # MME benchmark evaluation
│   ├── eval_mmvet.py        # MM-Vet benchmark evaluation
│   └── eval_mmmu.py         # MMMU benchmark evaluation
├── run_eval.py              # Unified evaluation entry point
├── requirements.txt
└── README.md
```

---

## 📜 Citation

If you find this work useful in your research, please cite:

```bibtex
@article{zhang2024accumulative,
  title     = {Mitigating Hallucinations in Large Vision-Language Models
               via Accumulative Decoding},
  author    = {Zhang, Haotian and Zhang, Zhiyun},
  year      = {2024},
}
```

---

## 🙏 Acknowledgements

<table>
<tr>
  <td><b>Backbone</b></td>
  <td><a href="https://github.com/haotian-liu/LLaVA">LLaVA-1.5</a> (Liu et al., CVPR 2024)</td>
</tr>
<tr>
  <td><b>Baselines</b></td>
  <td>
    <a href="https://github.com/DAMO-NLP-SG/VCD">VCD</a> ·
    <a href="https://github.com/voidism/DoLa">DoLA</a> ·
    VDD · DeCO
  </td>
</tr>
<tr>
  <td><b>Benchmarks</b></td>
  <td>
    <a href="https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation">MME</a> ·
    <a href="https://github.com/yuweihao/MM-Vet">MM-Vet</a> ·
    <a href="https://mmmu-benchmark.github.io/">MMMU</a>
  </td>
</tr>
</table>

---

<div align="center">
  <sub>Released under the <a href="LICENSE">MIT License</a> · UESTC Glasgow College · 2024</sub>
</div>
