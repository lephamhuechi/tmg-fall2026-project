"""Vietnamese Banking Intent Classification - Gradio demo.

Pipeline (same as the training notebook):
    text -> Unicode NFC + whitespace cleanup -> underthesea word segmentation
         -> PhoBERT (fine-tuned) -> softmax(logits / temperature)
         -> if best in-scope probability < threshold: `out_of_scope`

Expected folder layout (produced by the export cell of the notebook):
    app.py
    requirements.txt
    final_phobert/
        model_state.pt
        metadata.json
        tokenizer/

Run:  python app.py   (then open http://localhost:7860)
Override the artifact folder with:  ARTIFACT_DIR=/path/to/final_phobert python app.py
"""
import json
import os
import re
import unicodedata
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from underthesea import word_tokenize

# --------------------------------------------------------------------------
# Configuration (everything model-specific comes from metadata.json)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", BASE_DIR / "final_phobert"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

with (ARTIFACT_DIR / "metadata.json").open(encoding="utf-8") as f:
    META = json.load(f)

MODEL_NAME = META["model_name"]
MAX_LENGTH = int(META["max_length"])
ID2LABEL = {int(k): v for k, v in META["id2label"].items()}
NUM_LABELS = len(ID2LABEL)
OOS_ID = int(META["oos_id"])
TEMPERATURE = float(META["temperature"])
DEFAULT_THRESHOLD = float(META["oos_threshold"])
KNOWN_IDS = [i for i in range(NUM_LABELS) if i != OOS_ID]


# --------------------------------------------------------------------------
# Model (same architecture as in the notebook: PhoBERT + dropout + linear head,
# CLS pooling). The encoder is built from the config only; the fine-tuned
# weights come from model_state.pt, so the 540 MB base checkpoint is not downloaded.
# --------------------------------------------------------------------------
class PhoBERTIntentClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**kwargs).last_hidden_state
        return self.classifier(self.dropout(hidden[:, 0, :]))


def load_model():
    model = PhoBERTIntentClassifier(MODEL_NAME, NUM_LABELS)
    state = torch.load(ARTIFACT_DIR / "model_state.pt", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # `position_ids` is a buffer that some transformers versions do not store; anything else is a real mismatch.
    problems = [k for k in (*missing, *unexpected) if "position_ids" not in k]
    if problems:
        raise RuntimeError(f"model_state.pt does not match the model architecture: {problems[:5]}")
    return model.to(DEVICE).eval()


TOKENIZER = AutoTokenizer.from_pretrained(ARTIFACT_DIR / "tokenizer")
MODEL = load_model()


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def normalize_text(text):
    text = unicodedata.normalize("NFC", text).strip()
    return re.sub(r"\s+", " ", text)


def segment_vietnamese(text):
    return word_tokenize(normalize_text(text), format="text")


@torch.inference_mode()
def predict_probs(text):
    """Temperature-scaled probabilities over all labels (including out_of_scope)."""
    enc = TOKENIZER(segment_vietnamese(text), truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    logits = MODEL(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        token_type_ids=enc.get("token_type_ids"),
    )
    return torch.softmax(logits / TEMPERATURE, dim=-1)[0].cpu().numpy()


def decide(probs, threshold):
    """Return (predicted_id, is_oos, best in-scope probability)."""
    best_known = max(KNOWN_IDS, key=lambda i: probs[i])
    max_known = float(probs[best_known])
    if max_known < threshold:
        return OOS_ID, True, max_known
    return best_known, False, max_known


def gradio_predict(text, top_k, oos_threshold):
    if not text or not text.strip():
        return "⚠️ Vui lòng nhập câu cần phân loại.", "", "", ""
    try:
        probs = predict_probs(text)
        top_indices = np.argsort(probs)[::-1][: int(top_k)]
        top_k_text = "\n".join(
            f"{rank}. `{ID2LABEL[int(idx)]}` — **{probs[idx] * 100:.2f}%**"
            for rank, idx in enumerate(top_indices, 1)
        )

        threshold = float(oos_threshold)
        predicted_id, is_oos, max_known = decide(probs, threshold)
        predicted_intent = ID2LABEL[predicted_id]
        confidence = float(probs[predicted_id])

        if is_oos:
            intent_md = f"⚠️ OUT OF SCOPE\n\n**Intent:** `{predicted_intent}`"
            warning_md = (
                f"⚠️ **Cảnh báo OOS:** câu nhập vào có xác suất cao nhất trong các intent hỗ trợ "
                f"chỉ là **{max_known * 100:.2f}%**, thấp hơn ngưỡng **{threshold * 100:.2f}%**."
            )
        else:
            intent_md = f"### Predicted Intent\n\n**`{predicted_intent}`**"
            warning_md = "✅ Câu hỏi nằm trong phạm vi các intent được hỗ trợ."

        confidence_md = f"### Confidence\n\n**{confidence * 100:.2f}%**"
        return intent_md, confidence_md, top_k_text, warning_md
    except Exception as e:  # show the error in the UI instead of crashing the app
        return "❌ Prediction Error", "", "", f"```text\n{type(e).__name__}: {e}\n```"


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
EXAMPLES = [
    ["Hạn mức thẻ tín dụng của tôi là bao nhiêu?"],
    ["Tôi muốn khóa thẻ gấp"],
    ["Tôi muốn kiểm tra số dư tài khoản"],
    ["Tôi muốn đổi mã PIN"],
    ["Tôi muốn báo giao dịch gian lận"],
    ["Thời tiết hôm nay thế nào?"],
]

with gr.Blocks(title="Vietnamese Banking Intent Classification") as demo:
    gr.Markdown(
        """
        # Vietnamese Banking Intent Classification

        **PhoBERT + Temperature Scaling + OOS Detection**

        Nhập một câu hỏi/yêu cầu bằng tiếng Việt để hệ thống dự đoán **Banking Intent**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Input")
            input_text = gr.Textbox(
                label="Câu hỏi / Yêu cầu",
                placeholder="Ví dụ: Hạn mức thẻ tín dụng của tôi là bao nhiêu?",
                lines=5,
            )
            top_k = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top-K")
            oos_threshold = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=min(max(DEFAULT_THRESHOLD, 0.05), 0.95),
                step=0.01,
                label="OOS Threshold",
                info="Ngưỡng được tối ưu trên validation set; có thể điều chỉnh để demo.",
            )
            predict_btn = gr.Button("Dự đoán / Tra cứu", variant="primary")
            gr.Markdown("### Ví dụ")
            gr.Examples(examples=EXAMPLES, inputs=input_text)

        with gr.Column(scale=1):
            gr.Markdown("## Prediction")
            intent_output = gr.Markdown("Chưa có kết quả.")
            confidence_output = gr.Markdown()
            gr.Markdown("### Top-K Intents")
            top_k_output = gr.Markdown()
            warning_output = gr.Markdown()

    outputs = [intent_output, confidence_output, top_k_output, warning_output]
    predict_btn.click(fn=gradio_predict, inputs=[input_text, top_k, oos_threshold], outputs=outputs)
    input_text.submit(fn=gradio_predict, inputs=[input_text, top_k, oos_threshold], outputs=outputs)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
