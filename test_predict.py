"""Quick test of the fixed preprocessing pipeline."""
import numpy as np
from src.predict import preprocess_image, predict_digit, load_model

# Simulate REAL st_canvas output (alpha=255 everywhere)
canvas = np.zeros((280, 280, 4), dtype=np.uint8)
canvas[:, :, 3] = 255  # Alpha is 255 everywhere

# Draw digit '1' (vertical line)
canvas[40:240, 130:150, 0] = 255
canvas[40:240, 130:150, 1] = 255
canvas[40:240, 130:150, 2] = 255

processed = preprocess_image(canvas)
print("Processed shape:", processed.shape)
print("Processed max:", round(float(processed.max()), 4))
print("Processed mean:", round(float(processed.mean()), 4))

model = load_model()
result = predict_digit(canvas, model)
digit = result["digit"]
conf = result["confidence"]
print(f"Predicted: {digit} (confidence: {conf:.1f}%)")

# Show top 3 probabilities
probs = list(enumerate(result["probabilities"]))
probs.sort(key=lambda x: -x[1])
for d, p in probs[:3]:
    print(f"  Digit {d}: {p*100:.1f}%")
