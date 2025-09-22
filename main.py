from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn
import os
from datetime import datetime
import pandas as pd
import json
import logging

from langgraph_workflow import PositioningWorkflow
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PMM Assistant", description="Positioning & Messaging Assistant")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize the LangGraph workflow
workflow = PositioningWorkflow()

# Data models
class UserResponse(BaseModel):
    step: int
    question_id: str
    response: str
    session_id: str

class WorkflowState(BaseModel):
    session_id: str
    current_step: int
    completed_steps: List[int]
    responses: Dict[str, Any]
    is_complete: bool = False

# In-memory storage (in production, use a database)
workflow_states: Dict[str, WorkflowState] = {}

# Workflow steps definition
WORKFLOW_STEPS = [
    "Context Gathering & Planning",
    "Customer Understanding & Persona Development", 
    "Stakeholder & Customer Interviews",
    "Category & Competitive Positioning",
    "Feature/Benefit Translation (\"Product Legos\")",
    "Positioning Statement Creation",
    "Positioning Workshop Facilitation",
    "Messaging Framework & Brand Essence",
    "Proof Points & Narrative",
    "Asset Generation & Application",
    "Asset Inventory & Message Testing"
]

# Step 1 questions
STEP1_QUESTIONS = [
    {
        "id": "company_name",
        "question": "What is your company name?",
        "type": "text",
        "required": True
    },
    {
        "id": "company_description", 
        "question": "Please provide a brief description of what your company does.",
        "type": "textarea",
        "required": True
    },
    {
        "id": "customer_description",
        "question": "Please provide a brief description of who your customers are.",
        "type": "textarea", 
        "required": True
    },
    {
        "id": "positioning_experience",
        "question": "Are you doing the positioning and messaging exercise for the first time, or are you repositioning an existing brand?",
        "type": "select",
        "options": ["First time", "Repositioning existing brand"],
        "required": True
    },
    {
        "id": "company_scope",
        "question": "Are you doing this exercise for the whole company or a specific segment/product?",
        "type": "select",
        "options": ["Whole company", "Specific segment/product"],
        "required": True
    }
]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Main application page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "workflow_steps": WORKFLOW_STEPS,
        "current_step": 1,
        "step1_questions": STEP1_QUESTIONS
    })

@app.post("/api/start-workflow")
async def start_workflow(request: Request):
    """Start a new workflow session"""
    data = await request.json()
    session_id = data.get("session_id")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    
    workflow_states[session_id] = WorkflowState(
        session_id=session_id,
        current_step=1,
        completed_steps=[],
        responses={}
    )
    return {"status": "started", "current_step": 1, "session_id": session_id}

@app.get("/api/workflow-state/{session_id}")
async def get_workflow_state(session_id: str):
    """Get current workflow state"""
    if session_id not in workflow_states:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = workflow_states[session_id]
    return {
        "current_step": state.current_step,
        "completed_steps": state.completed_steps,
        "responses": state.responses,
        "is_complete": state.is_complete
    }

@app.post("/api/submit-response")
async def submit_response(request: Request):
    """Submit a user response and get next question or move to next step"""
    data = await request.json()
    
    session_id = data.get("session_id")
    step = data.get("step")
    question_id = data.get("question_id")
    response = data.get("response")
    
    if not all([session_id, step, question_id, response]):
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    # If session doesn't exist, recreate it
    if session_id not in workflow_states:
        logger.warning(f"Session {session_id} not found, recreating...")
        workflow_states[session_id] = WorkflowState(
            session_id=session_id,
            current_step=1,
            completed_steps=[],
            responses={}
        )
    
    state = workflow_states[session_id]
    
    # Store the response
    state.responses[question_id] = response
    
    # Check if current step is complete
    if step == 1:
        # Check if all Step 1 questions are answered
        step1_question_ids = [q["id"] for q in STEP1_QUESTIONS]
        answered_questions = [qid for qid in step1_question_ids if qid in state.responses]
        
        if len(answered_questions) == len(step1_question_ids):
            # Step 1 complete, move to step 2
            state.completed_steps.append(1)
            state.current_step = 2
            
            # Save responses to CSV
            await save_responses_to_csv(session_id, state.responses)
            
            return {
                "status": "step_complete",
                "next_step": 2,
                "message": "Step 1 completed! Moving to Step 2: Customer Understanding & Persona Development"
            }
        else:
            # Still more questions in current step
            remaining_questions = [q for q in STEP1_QUESTIONS if q["id"] not in answered_questions]
            return {
                "status": "continue_step",
                "next_question": remaining_questions[0] if remaining_questions else None,
                "remaining_count": len(remaining_questions)
            }
    
    return {"status": "response_saved"}

@app.post("/api/validate-response")
async def validate_response(request: Request):
    """Validate user response using LangGraph"""
    data = await request.json()
    
    session_id = data.get("session_id")
    message = data.get("message")
    question = data.get("question")
    question_type = data.get("question_type")
    
    if not all([session_id, message, question]):
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    try:
        # Use LangGraph workflow to validate response
        validation_response = await workflow.validate_response(
            message=message,
            question=question,
            question_type=question_type,
            session_id=session_id
        )
        
        return {"is_valid": validation_response}
    except Exception as e:
        # Fallback validation
        is_valid = (
            len(message.strip()) > 2 and 
            message.strip().lower() not in ['ok', 'yes', 'no', ',', '.', '...'] and
            not message.strip().isdigit()
        )
        return {"is_valid": is_valid}

@app.post("/api/generate-plan")
async def generate_pmm_plan(request: Request):
    """Generate a personalized PMM plan based on Step 1 responses"""
    data = await request.json()
    
    session_id = data.get("session_id")
    responses = data.get("responses", {})
    
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    
    try:
        # Generate personalized plan using LangGraph workflow
        plan = await workflow.generate_pmm_plan(responses, session_id)
        
        return {"plan": plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating plan: {str(e)}")

@app.post("/api/chat")
async def chat_with_workflow(request: Request):
    """Chat endpoint for LangGraph workflow"""
    data = await request.json()
    session_id = data.get("session_id")
    message = data.get("message")
    
    if not session_id or not message:
        raise HTTPException(status_code=400, detail="session_id and message are required")
    
    # If session doesn't exist, recreate it
    if session_id not in workflow_states:
        logger.warning(f"Session {session_id} not found, recreating...")
        workflow_states[session_id] = WorkflowState(
            session_id=session_id,
            current_step=1,
            completed_steps=[],
            responses={}
        )
    
    state = workflow_states[session_id]
    
    # Process with LangGraph workflow
    try:
        response = await workflow.process_message(
            message=message,
            session_id=session_id,
            current_step=state.current_step,
            previous_responses=state.responses
        )
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow error: {str(e)}")

async def save_responses_to_csv(session_id: str, responses: Dict[str, Any]):
    """Save user responses to CSV file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"responses_{session_id}_{timestamp}.csv"
    
    # Create DataFrame
    df_data = []
    for question_id, response in responses.items():
        df_data.append({
            "session_id": session_id,
            "question_id": question_id,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv(filename, index=False)
    
    return filename

@app.get("/api/workflow-steps")
async def get_workflow_steps():
    """Get all workflow steps"""
    return {"steps": WORKFLOW_STEPS}

@app.get("/api/step-questions/{step_number}")
async def get_step_questions(step_number: int):
    """Get questions for a specific step"""
    if step_number == 1:
        return {"questions": STEP1_QUESTIONS}
    else:
        return {"questions": [], "message": f"Step {step_number} questions not yet implemented"}

@app.post("/api/upload-persona-document")
async def upload_persona_document(request: Request):
    """Handle persona document upload for Step 2"""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        filename = data.get("filename")
        
        if not session_id or not filename:
            raise HTTPException(status_code=400, detail="Session ID and filename are required")
        
        # For now, just acknowledge the upload (in a real app, you'd save the file)
        logger.info(f"Persona document uploaded for session {session_id}: {filename}")
        
        return {"message": "Document uploaded successfully", "filename": filename}
        
    except Exception as e:
        logger.error(f"Error uploading persona document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/get-customer-research-todo")
async def get_customer_research_todo():
    """Generate customer research todo list for Step 2"""
    try:
        todo_list = """
## 📋 **Customer Research Action Plan**

To identify your ICP, buyers, influencers, and users, here's what you need to do:

### 1. **Define Your Ideal Customer Profile (ICP)**
- Identify demographic characteristics (age, location, company size, industry)
- Determine firmographic details (revenue, employee count, technology stack)
- Understand psychographic traits (goals, challenges, pain points)

### 2. **Map Your Buyer Personas**
- Identify decision-makers vs. end-users
- Understand their roles, responsibilities, and influence levels
- Map their buying journey and decision criteria

### 3. **Identify Key Influencers**
- Find industry thought leaders and experts
- Identify internal champions and advocates
- Map influencer networks and communities

### 4. **Research User Personas**
- Understand end-user needs and behaviors
- Identify usage patterns and preferences
- Map user journey and touchpoints

### 5. **Conduct Market Research**
- Analyze competitor customer bases
- Study industry reports and surveys
- Gather insights from customer interviews

Would you like me to conduct automated competitor research to help identify your ICP, buyers, influencers, and users?
        """
        
        return {"todo_list": todo_list}
        
    except Exception as e:
        logger.error(f"Error generating todo list: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conduct-competitor-research")
async def conduct_competitor_research(request: Request):
    """Conduct automated competitor research using GPT-5-nano"""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
        
        # Get session state to access user responses
        if session_id not in workflow_states:
            raise HTTPException(status_code=404, detail="Session not found")
        
        state = workflow_states[session_id]
        
        # Use LangGraph to conduct research with actual user responses
        research_results = await workflow.conduct_competitor_research_with_responses(
            session_id, state.responses
        )
        
        return {"research_results": research_results}
        
    except Exception as e:
        logger.error(f"Error conducting competitor research: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Validate configuration
    Config.validate()
    
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
