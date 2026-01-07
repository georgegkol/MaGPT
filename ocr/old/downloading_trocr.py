from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_ID = "microsoft/trocr-base-handwritten"
SAVE_DIR = "models/trocr-handwritten"

processor = TrOCRProcessor.from_pretrained(MODEL_ID)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

processor.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

print("Model saved locally to:", SAVE_DIR)