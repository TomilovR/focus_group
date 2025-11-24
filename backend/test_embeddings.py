from openai import OpenAI
import sys

def test_embeddings():
    print("Testing embeddings on LM Studio...")
    client = OpenAI(base_url="http://192.168.0.113:1234/v1", api_key="lm-studio")
    
    try:
        response = client.embeddings.create(
            model="local-model",
            input="The food was delicious and the waiter..."
        )
        print(f"Success! Embedding length: {len(response.data[0].embedding)}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        print("Local server might not support embeddings or model is not loaded.")
        return False

if __name__ == "__main__":
    test_embeddings()
