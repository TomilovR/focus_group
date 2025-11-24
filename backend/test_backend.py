from backend.models import EmailDraft
from backend.simulation import Simulator
import json

def test_simulation():
    draft = EmailDraft(
        subject="Test Subject",
        body="This is a test body.",
        cta="Click here",
        audience="Tech"
    )
    
    sim = Simulator()
    result = sim.run_simulation(draft)
    
    print(f"Simulation ID: {result.id}")
    print(f"Metrics: {result.metrics}")
    print(f"Responses: {len(result.responses)}")
    print(f"First Response: {result.responses[0]}")
    
    assert len(result.responses) == 10
    assert result.metrics.openRate >= 0
    
    print("Test Passed!")

if __name__ == "__main__":
    test_simulation()
