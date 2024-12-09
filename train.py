from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Create a very small, focused dataset
debug_data = [
    ("t", "the"),
    ("th", "the"),
    ("qk", "quick"),
    ("brwn", "brown"),
    ("fx", "fox"),
    ("t qk", "the quick"),
    ("qk brwn", "quick brown"),
    ("t qk brwn", "the quick brown"),
    ("t qk brwn fx", "the quick brown fox")
] * 100  # Repeat to have enough examples

class DebugDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        short, full = self.pairs[idx]
        
        # Explicitly add start and end tokens
        source_text = f"{short} </s>"
        target_text = f"{full} </s>"
        
        # Tokenize source
        source = self.tokenizer(
            source_text,
            padding='max_length',
            max_length=32,
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target
        target = self.tokenizer(
            target_text,
            padding='max_length',
            max_length=32,
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': source['input_ids'].squeeze(),
            'attention_mask': source['attention_mask'].squeeze(),
            'labels': target['input_ids'].squeeze()
        }

def train_debug_model():
    # Initialize model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    
    # Create dataset
    dataset = DebugDataset(debug_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    # Train for a few epochs
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
    
    return model, tokenizer

def test_model(model, tokenizer, test_inputs):
    model.eval()
    for text in test_inputs:
        input_ids = tokenizer(
            f"{text}", 
            return_tensors="pt", 
            max_length=32, 
            truncation=True
        ).input_ids
        
        outputs = model.generate(
            input_ids,
            max_length=32,
            num_beams=5,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
        )
        
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {text}")
        print(f"Output: {decoded}\n")

if __name__ == "__main__":
    # Train model
    model, tokenizer = train_debug_model()
    
    # Test cases
    test_inputs = [
        "t qk brwn fx",
        "t",
        "qk",
        "brwn",
        "t qk"
    ]
    
    print("\nTesting model outputs:")
    test_model(model, tokenizer, test_inputs)