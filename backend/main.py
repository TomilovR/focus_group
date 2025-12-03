from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from models import EmailDraft, SimulationResult
from simulation import Simulator
from database import get_db, AudienceModel, SimulationModel, PersonaModel, ResponseModel
from config import CORS_ORIGINS, logger
from validators import EmailValidator, ValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import traceback

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Email AI Predictor API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS with configurable origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

simulator = Simulator()

from fastapi.responses import StreamingResponse
import json

@app.post("/api/simulate")
@limiter.limit("10/minute")  # Rate limit: 10 simulations per minute
async def simulate_email(request: Request, draft: EmailDraft, db: Session = Depends(get_db)):
    logger.info(f"Simulation request received - Subject: {draft.subject[:50]}...")
    
    # Validate and sanitize inputs
    try:
        validated_subject = EmailValidator.validate_subject(draft.subject)
        validated_body = EmailValidator.validate_body(draft.body)
        validated_cta = EmailValidator.validate_cta(draft.cta)
        validated_audience = EmailValidator.validate_audience(draft.audience)
        
        # Update draft with validated data
        draft.subject = validated_subject
        draft.body = validated_body
        draft.cta = validated_cta
        draft.audience = validated_audience
        
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected validation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal validation error")
    
    def event_generator():
        final_result_data = None
        try:
            logger.info(f"Starting simulation for audience: {draft.audience}")
            for event in simulator.run_simulation_stream(draft):
                yield json.dumps(event) + "\n"
                if event["type"] == "result":
                    final_result_data = event["data"]
            
            # Save to DB after simulation is done
            if final_result_data:
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
                logger.info(f"Simulation {sim_model.id} saved to database")
                
        except Exception as e:
            logger.error(f"Error during simulation stream: {str(e)}")
            logger.error(traceback.format_exc())
            db.rollback()
            yield json.dumps({"type": "error", "message": "Simulation failed. Please try again."}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.get("/api/audiences")
@limiter.limit("30/minute")
async def get_audiences(request: Request, db: Session = Depends(get_db)):
    try:
        logger.info("Fetching audiences")
        audiences = db.query(AudienceModel).all()
        return [
            {
                "id": a.id,
                "name": a.name,
                "type": a.type,
                "size": len(a.personas),
                "lastUpdated": "Just now",
                "personas": [p.to_dict() for p in a.personas]
            }
            for a in audiences
        ]
    except Exception as e:
        logger.error(f"Error fetching audiences: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch audiences")

@app.get("/api/history")
@limiter.limit("30/minute")
async def get_history(request: Request, db: Session = Depends(get_db)):
    try:
        logger.info("Fetching simulation history")
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
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch history")

@app.delete("/api/history")
@limiter.limit("5/minute")
async def clear_history(request: Request, db: Session = Depends(get_db)):
    try:
        logger.warning("Clearing simulation history")
        db.query(ResponseModel).delete()
        db.query(SimulationModel).delete()
        db.commit()
        logger.info("History cleared successfully")
        return {"status": "cleared"}
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to clear history")

@app.get("/api/history/{sim_id}")
@limiter.limit("30/minute")
async def get_simulation_detail(request: Request, sim_id: str, db: Session = Depends(get_db)):
    try:
        logger.info(f"Fetching simulation detail: {sim_id}")
        sim = db.query(SimulationModel).filter(SimulationModel.id == sim_id).first()
        if not sim:
            logger.warning(f"Simulation not found: {sim_id}")
            raise HTTPException(status_code=404, detail="Simulation not found")
        
        responses = []
        for r in sim.responses:
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching simulation detail: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch simulation details")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Email AI Predictor API")
    uvicorn.run(app, host="0.0.0.0", port=8000)
