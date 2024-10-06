from transformers import T5ForConditionalGeneration
import torch

model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")

num_special_tokens = 3
# Model has 3 special tokens which take up the input ids 0,1,2 of ByT5.
# => Need to shift utf-8 character encodings by 3 before passing ids to model.

input_ids = torch.tensor([list("Life is like a box of chocolates.".encode("utf-8"))]) + num_special_tokens

labels = torch.tensor([list("La vie est comme une boîte de chocolat.".encode("utf-8"))]) + num_special_tokens

loss = model(input_ids, labels=labels).loss
print(loss.item())

# Generate output predictions
# Note: You may want to specify parameters like max_length and num_beams for better results
generated_ids = model.generate(input_ids, max_length=50)

# Decode the generated ids back to text
predicted_output = ''.join(chr(byte) for byte in generated_ids[0].tolist() if byte >= num_special_tokens)  # Exclude special tokens
print("Predicted Output:", predicted_output)