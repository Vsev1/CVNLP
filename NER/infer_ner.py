import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_DIR = "ner_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def predict_sentence(model, sentence, tokenizer, schema, device):
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    tags = [schema[p] for p in predictions]

    for token, tag in zip(tokens, tags):
        print(f"{token:15} -> {tag}")


def main():
    print("\nPrediction on Example 1:")
    predict_sentence(model, "I think Mount Everest looks beautiful", tokenizer, schema, device)

    print("\nPrediction on Example 2:")
    predict_sentence(model, "She came to France yesterday", tokenizer, schema, device)


if __name__ == '__main__':
    main()