from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from models import EmailDraft, SimulationResult
from simulation import Simulator
from database import get_db, AudienceModel, SimulationModel, PersonaModel, ResponseModel

app = FastAPI(title="Email AI Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

simulator = Simulator()

from fastapi.responses import StreamingResponse
import json

@app.post("/api/simulate")
async def simulate_email(draft: EmailDraft, db: Session = Depends(get_db)):
    def event_generator():
        final_result_data = None
        try:
            for event in simulator.run_simulation_stream(draft):
                yield json.dumps(event) + "\n"
                if event["type"] == "result":
                    final_result_data = event["data"]
            
            # Save to DB after simulation is done
            if final_result_data:
                # Reconstruct SimulationResult object from dict for easier access if needed, 
                # or just use the dict directly to populate models.
                # The event["data"] is the .dict() of SimulationResult.
                
                res_data = final_result_data
                
                sim_model = SimulationModel(
                    id=res_data['id'],
                    timestamp=res_data['timestamp'],
                    subject=draft.subject,
                    body=draft.body,
                    cta=draft.cta,
                    audience_target=draft.audience,
                    metrics=res_data['metrics'],
                    insights=res_data['insights']
                )
                db.add(sim_model)
                db.flush()
                
                for r in res_data['responses']:
                    # r is a dict here because we serialized it
                    # We need to get persona_id. In the dict, persona is a dict too.
                    persona_id = r['persona']['id']
                    
                    resp_model = ResponseModel(
                        simulation_id=sim_model.id,
                        persona_id=persona_id,
                        action=r['action'],
                        sentiment=r['sentiment'],
                        comment=r['comment'],
                        detailed_reasoning=r['detailedReasoning']
                    )
                    db.add(resp_model)
                    
                db.commit()
                print("Simulation saved to DB.")
                
        except Exception as e:
            print(f"Error during simulation stream: {e}")
            db.rollback()
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.get("/api/audiences")
async def get_audiences(db: Session = Depends(get_db)):
    audiences = db.query(AudienceModel).all()
    return [
        {
            "id": a.id,
            "name": a.name,
            "type": a.type,
            "size": len(a.personas),
            "lastUpdated": "Just now", # Placeholder
            "personas": [p.to_dict() for p in a.personas] # Return all personas
        }
        for a in audiences
    ]

@app.get("/api/history")
async def get_history(db: Session = Depends(get_db)):
    simulations = db.query(SimulationModel).order_by(SimulationModel.timestamp.desc()).all()
    return [
        {
            "id": s.id,
            "timestamp": s.timestamp,
            "subject": s.subject,
            "metrics": s.metrics,
            "audience": s.audience_target
        }
        for s in simulations
    ]

@app.delete("/api/history")
async def clear_history(db: Session = Depends(get_db)):
    try:
        # Delete all responses first to avoid FK constraint issues if cascade is not set
        db.query(ResponseModel).delete()
        db.query(SimulationModel).delete()
        db.commit()
        return {"status": "cleared"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{sim_id}")
async def get_simulation_detail(sim_id: str, db: Session = Depends(get_db)):
    sim = db.query(SimulationModel).filter(SimulationModel.id == sim_id).first()
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    # Reconstruct SimulationResult
    # We need to fetch responses and their personas
    responses = []
    for r in sim.responses:
        # Fetch persona manually or rely on relationship
        # r.persona is available via relationship
        p_model = r.persona
        if p_model:
            persona_obj = {
                "id": p_model.id,
                "name": p_model.name,
                "role": p_model.role,
                "company": p_model.company,
                "avatar": p_model.avatar,
                "psychographics": p_model.psychographics,
                "pastBehavior": p_model.past_behavior
            }
        else:
            # Fallback if persona deleted
            persona_obj = {
                "id": "unknown",
                "name": "Unknown",
                "role": "N/A",
                "company": "N/A",
                "avatar": "?",
                "psychographics": "",
                "pastBehavior": ""
            }
            
        responses.append({
            "persona": persona_obj,
            "action": r.action,
            "sentiment": r.sentiment,
            "comment": r.comment,
            "detailedReasoning": r.detailed_reasoning
        })

    return {
        "id": sim.id,
        "timestamp": sim.timestamp,
        "subject": sim.subject,
        "body": sim.body,
        "cta": sim.cta,
        "metrics": sim.metrics,
        "insights": sim.insights,
        "responses": responses
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}
