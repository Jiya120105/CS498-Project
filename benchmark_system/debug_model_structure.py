"""
Quick script to inspect SmolVLM model structure and find where the layers are.
"""

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

print("Loading SmolVLM...")
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    _attn_implementation="eager"
)

print(f"\nModel type: {type(model)}")
print(f"\nTop-level attributes:")
for attr in sorted([x for x in dir(model) if not x.startswith('_')])[:30]:
    print(f"  {attr}")

# Check model.model
if hasattr(model, 'model'):
    print(f"\nmodel.model type: {type(model.model)}")
    print(f"model.model attributes:")
    for attr in sorted([x for x in dir(model.model) if not x.startswith('_')])[:30]:
        print(f"  {attr}")

    # Check for layers
    if hasattr(model.model, 'layers'):
        print(f"\n✓ Found layers at model.model.layers!")
        print(f"  Number of layers: {len(model.model.layers)}")
        print(f"  Layer type: {type(model.model.layers[0])}")
    else:
        print(f"\n✗ No 'layers' attribute at model.model")

        # Check text_model inside model.model
        if hasattr(model.model, 'text_model'):
            print(f"\n model.model.text_model type: {type(model.model.text_model)}")
            print(f" model.model.text_model attributes:")
            for attr in sorted([x for x in dir(model.model.text_model) if not x.startswith('_')])[:30]:
                print(f"    {attr}")

            if hasattr(model.model.text_model, 'layers'):
                print(f"\n  ✓ Found layers at model.model.text_model.layers!")
                print(f"    Number of layers: {len(model.model.text_model.layers)}")
                print(f"    Layer 0 type: {type(model.model.text_model.layers[0])}")

        # Check vision_model
        if hasattr(model.model, 'vision_model'):
            print(f"\n model.model.vision_model exists")
            if hasattr(model.model.vision_model, 'encoder'):
                if hasattr(model.model.vision_model.encoder, 'layers'):
                    print(f"  ✓ Found vision layers at model.model.vision_model.encoder.layers")
                    print(f"    Number of vision layers: {len(model.model.vision_model.encoder.layers)}")

# Check text_model
if hasattr(model, 'text_model'):
    print(f"\nmodel.text_model type: {type(model.text_model)}")
    if hasattr(model.text_model, 'layers'):
        print(f"✓ Found layers at model.text_model.layers!")
        print(f"  Number of layers: {len(model.text_model.layers)}")

# Check language_model
if hasattr(model, 'language_model'):
    print(f"\nmodel.language_model type: {type(model.language_model)}")
    if hasattr(model.language_model, 'model'):
        if hasattr(model.language_model.model, 'layers'):
            print(f"✓ Found layers at model.language_model.model.layers!")
            print(f"  Number of layers: {len(model.language_model.model.layers)}")

print("\nDone!")
