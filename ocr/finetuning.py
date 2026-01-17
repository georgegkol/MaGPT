import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = Path("./data/lines_for_transcription")
CSV_PATH = DATA_DIR / "transcriptions.csv"

MODEL_NAME = "fhswf/TrOCR_german_handwritten"
OUTPUT_DIR = Path("./models/trocr_finetuned")

TEST_PAGE = "page004"   # held-out page

BATCH_SIZE = 1          # memory-friendly
EPOCHS = 14
LR = 5e-6

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


from torch.utils.data import DataLoader

def collate_fn(batch):
    # batch is a list of dicts: {"pixel_values": ..., "labels": ...}
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(CSV_PATH)
df["page"] = df["image"].apply(lambda x: x.split("/")[0])

train_df = df[df["page"] != TEST_PAGE].reset_index(drop=True)
test_df  = df[df["page"] == TEST_PAGE].reset_index(drop=True)

print(f"Train lines: {len(train_df)}")
print(f"Test lines:  {len(test_df)}")
assert len(test_df) > 0, "Test set is empty!"

# -----------------------------
# DATASET CLASS
# -----------------------------
class OCRDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = DATA_DIR / row["image"]
        img = Image.open(img_path).convert("RGB")
        labels = row["text"]

        # pixel_values and labels for model
        pixel_values = self.processor(img, return_tensors="pt").pixel_values.squeeze()
        with self.processor.as_target_processor():
            labels_enc = self.processor(labels, return_tensors="pt").input_ids.squeeze()
        return {"pixel_values": pixel_values, "labels": labels_enc}

# -----------------------------
# LOAD MODEL + PROCESSOR
# -----------------------------
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# Freeze encoder to save memory
for param in model.encoder.parameters():
    param.requires_grad = False

# -----------------------------
# DATASETS
# -----------------------------
train_dataset = OCRDataset(train_df, processor)
eval_dataset  = OCRDataset(test_df, processor)

# -----------------------------
# TRAINING ARGUMENTS
# -----------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=BATCH_SIZE,   # low memory
    per_device_eval_batch_size=1,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    predict_with_generate=True,
    logging_steps=5,
    evaluation_strategy="steps",
    eval_steps=len(train_dataset),      # once per epoch
    save_strategy="steps",
    save_steps=len(train_dataset),      # save once per epoch
    save_total_limit=1,                 # only keep last checkpoint
    remove_unused_columns=False,
    report_to="none",
)


# -----------------------------
# TRAINER
# -----------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
    data_collator=collate_fn
)

trainer.train()
