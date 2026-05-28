# Test auto-loading functionality
from wyze_embedding import (
    load_person_model,
    load_pet_model,
    list_available_models
)

# List available models
print("Available models:", list_available_models())

# Load and test models
person_model = load_person_model()
pet_model = load_pet_model()

print("✅ Installation successful - models loaded correctly!")