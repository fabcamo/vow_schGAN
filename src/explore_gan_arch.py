from pathlib import Path
from tensorflow.keras.models import load_model

# Load your model
model_path = Path(r"D:\schemaGAN\h5\schemaGAN.h5")
model = load_model(model_path, compile=False)

# Print summary
print("=" * 60)
print("SchemaGAN Model Architecture")
print("=" * 60)
model.summary()

# Specifically check for Dropout layers
print("\n" + "=" * 60)
print("Dropout Layer Check:")
print("=" * 60)
has_dropout = False
for i, layer in enumerate(model.layers):
    if "dropout" in layer.name.lower() or "Dropout" in str(type(layer)):
        print(f"Layer {i}: {layer.name} - {type(layer).__name__}")
        if hasattr(layer, "rate"):
            print(f"  Dropout rate: {layer.rate}")
        has_dropout = True

if not has_dropout:
    print("No Dropout layers found in model")
else:
    print(f"\n✓ Model has Dropout layers!")
