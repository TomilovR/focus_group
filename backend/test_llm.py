from backend.llm_service import OpenAILLM
import sys

def test_connection():
    print("Testing connection to LM Studio...")
    try:
        llm = OpenAILLM()
        response = llm.predict("Hello! Are you ready to simulate email recipients?")
        print(f"Response from LLM:\n{response}")
        
        if response and response != "{}":
            print("\nSUCCESS: Connected to LM Studio!")
        else:
            print("\nFAILURE: Received empty response.")
            
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Make sure 'openai' is installed: pip install openai")

if __name__ == "__main__":
    test_connection()
