from google import genai
from google.genai import types

# Initialize the client for Vertex AI
# It will automatically use the VM's Service Account credentials
client = genai.Client(
    vertexai=True,
    project="ai-datascience-354723",  # Replace with your project ID
    location="global"      # Or your preferred region
)

def test_connection():
    print("--- Querying Gemini via Vertex AI ---")
    try:
        response = client.models.generate_content(
            # model="gemini-3.1-flash-lite-preview",
            model="gemini-2.5-pro",
            contents="Confirming connection: Tell me a one-sentence fun fact about cloud computing."
        )
        print(f"Success! Response: \n{response.text}")
    except Exception as e:
        print(f"Connection failed. Error: {e}")

if __name__ == "__main__":

    client = genai.Client(vertexai=True, project="ai-datascience-354723", location="us-central1")

    print("Available Gemini Models on Vertex AI:")
    for model in client.models.list():
        # Filter for Gemini models to keep the list clean
        if "gemini" in model.name:
            print(f"ID: {model.name:30} | Display Name: {model.display_name}")

    test_connection()