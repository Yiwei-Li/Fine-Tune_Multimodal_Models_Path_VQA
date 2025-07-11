# Fine-Tuning Multimodal Models for Pathology Image Q&A

This repository presents a series of **multimodal vision-language experiments** in the field of **Pathology Visual Question Answering (Path-VQA)**. The project evaluates both general-purpose and domain-specific model setups, comparing baseline performance with LoRA fine-tuning and custom architecture integration.

<br>



## Dataset

- Dataset used: [flaviagiammarino/path-vqa](https://huggingface.co/datasets/flaviagiammarino/path-vqa)
- Originally contains ~20,000 training image-question-answer data  spanning medical diagrams, pathology, fetal development, and anatomy. This project randomly selected 8,000 training samples due to hardware limitations.
- Cleaned and preprocessed into a VQA-compatible format for model fine-tuning.



## Models Used

| Component | Hugging Face Model |
|-|-|
| BLIP Base VQA | [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base) |
| Image Encoder | [microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) |
| Language Decoder | [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) |



## Results Overview

### 1. **BLIP Base (Zero-Shot)**

- Out-of-the-box BLIP model on pathology images.
- Often fails to capture domain-specific terminology.
- Struggles with image semantics in pathology and diagrams.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Sample Output:**  
<img src=".\misc\base_model_inference.png" alt="BLIP Zero-Shot" width="600"/>


### 2. **BLIP Fine-Tuned on Medical VQA**

- Fine-tuned using LoRA on Path-VQA data for 3 epochs.
- Shows significant improvement on yes/no questions.
- Still exhibits hallucinations and domain misclassifications on open-ended or anatomy-specific questions.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Sample Output:**  
<img src=".\misc\fine_tune_inference_1.png" alt="BLIP Fine-Tuned" width="600"/>


### 3. **Custom BioMedCLIP + Mistral (Captioning)**

- Image Encoder: BioMedCLIP
- Text Decoder: Mistral 7B
- Fine-tuned using LoRA as an **image captioning task** (not Q&A) for 3 epochs on 4,000 samples.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Challenges Encountered**:
- Very low [METEOR](https://medium.com/on-being-ai/what-is-meteor-metric-for-evaluation-of-translation-with-explicit-ordering-45b49ac5ec70) scores (0.02).
- Likely causes:
  - Lack of a **vision-to-language projection head** between encoder and decoder.
  - Limited dataset size for large decoder models.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Sample Output:**  
<img src=".\misc\fine_tune_inference_2.png" alt="Custom Fine-Tuned" width="600"/>



## Metrics

| Model                | Exact Match | METEOR Score | 
|----------------------|-------------|-------|
| BLIP (Zero-Shot)     | 0.25 | 0.15 | 
| BLIP (Fine-Tuned)    | 0.46 | 0.23
| BioMedCLIP + Mistral | 0 | 0.02



## Limitations

### 1. Hardware Constraints
- All experiments were conducted on a laptop with a single RTX 5080 Mobile GPU (16 GB VRAM) and 32 GB system RAM.
- These constraints limited fine-tuning of larger models (e.g., BLIP-2, LLaVA) and capped the training data at 8,000 images.


### 2. Missing Vision-Language Adapter
- The custom BioMedCLIP + Mistral architecture lacked a projection layer to map vision embeddings into the language model’s input space.
- This likely caused misalignment, resulting in near-zero METEOR scores during captioning.



