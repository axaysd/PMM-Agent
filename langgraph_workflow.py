from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
import os
import logging
from config import Config

# Configure logging
logger = logging.getLogger(__name__)

class WorkflowState(BaseModel):
    messages: List[BaseMessage]
    current_step: int
    session_id: str
    user_responses: Dict[str, Any]
    follow_up_needed: bool = False
    current_question: Optional[str] = None

class PositioningWorkflow:
    def __init__(self):
        # Initialize OpenAI model (GPT-5-mini)
        self.model = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            api_key=Config.OPENAI_API_KEY,
            temperature=1.0  # GPT-5-mini only supports default temperature of 1
        )
        
        # Initialize checkpointer for state persistence
        self.checkpointer = InMemorySaver()
        
        # Build the workflow graph
        self.graph = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        def analyze_response(state: WorkflowState) -> Dict[str, Any]:
            """Analyze user response and determine if follow-up is needed"""
            if not state.messages:
                return {"follow_up_needed": False}
            
            last_message = state.messages[-1]
            if isinstance(last_message, HumanMessage):
                # Use LLM to analyze if response needs clarification
                analysis_prompt = f"""
                Analyze this user response for clarity and completeness. 
                Determine if follow-up questions are needed.
                
                Response: {last_message.content}
                Current Step: {state.current_step}
                
                Return 'NEEDS_FOLLOWUP' if the response is unclear, too brief, or needs clarification.
                Return 'COMPLETE' if the response is clear and sufficient.
                """
                
                analysis_response = self.model.invoke([HumanMessage(content=analysis_prompt)])
                
                needs_followup = "NEEDS_FOLLOWUP" in analysis_response.content
                
                return {
                    "follow_up_needed": needs_followup,
                    "analysis": analysis_response.content
                }
            
            return {"follow_up_needed": False}
        
        def generate_followup(state: WorkflowState) -> Dict[str, Any]:
            """Generate follow-up questions if needed"""
            if not state.messages:
                return {"messages": []}
            
            last_message = state.messages[-1]
            
            followup_prompt = f"""
            The user provided this response: "{last_message.content}"
            
            This response needs clarification. Generate 1-2 specific follow-up questions to get clearer, more detailed information.
            
            Be conversational and helpful. Ask for specific details, examples, or clarification.
            """
            
            followup_response = self.model.invoke([HumanMessage(content=followup_prompt)])
            
            return {
                "messages": [followup_response],
                "follow_up_needed": False
            }
        
        def generate_response(state: WorkflowState) -> Dict[str, Any]:
            """Generate appropriate response based on current step and context"""
            if not state.messages:
                return {"messages": []}
            
            last_message = state.messages[-1]
            
            # Create context-aware prompt based on current step
            context_prompt = self._get_context_prompt(state.current_step, state.user_responses)
            
            full_prompt = f"""
            {context_prompt}
            
            User message: {last_message.content}
            
            Provide a helpful, conversational response. If this is a response to a specific question, acknowledge it and ask the next question if appropriate.
            """
            
            response = self.model.invoke([HumanMessage(content=full_prompt)])
            
            return {"messages": [response]}
        
        def should_continue(state: WorkflowState) -> str:
            """Determine next action based on state"""
            if state.follow_up_needed:
                return "followup"
            else:
                return "response"
        
        # Build the graph
        builder = StateGraph(WorkflowState)
        
        # Add nodes
        builder.add_node("analyze", analyze_response)
        builder.add_node("followup", generate_followup)
        builder.add_node("response", generate_response)
        
        # Add edges
        builder.add_edge(START, "analyze")
        builder.add_conditional_edges(
            "analyze",
            should_continue,
            {
                "followup": "followup",
                "response": "response"
            }
        )
        builder.add_edge("followup", END)
        builder.add_edge("response", END)
        
        return builder.compile(checkpointer=self.checkpointer)
    
    def _get_context_prompt(self, current_step: int, user_responses: Dict[str, Any]) -> str:
        """Get context-aware prompt based on current step"""
        
        step_contexts = {
            1: """
            You are helping with Step 1: Context Gathering & Planning.
            
            The user is answering questions about:
            - Company name
            - Company description  
            - Customer description
            - Positioning experience (first time vs repositioning)
            - Company scope (whole company vs specific segment)
            
            Be conversational and encouraging. Ask follow-up questions if responses are unclear.
            """,
            2: """
            You are helping with Step 2: Customer Understanding & Persona Development.
            
            The user has been asked to upload persona documents or indicate they don't have them.
            If they mention uploading documents, acknowledge it and proceed with automated research.
            If they say they don't have personas, provide the research todo list and ask if they want automated research.
            If they want automated research, conduct competitor research to identify ICP, buyers, influencers, and users.
            
            Focus on understanding the target customers in detail through:
            - Document analysis (if uploaded)
            - Automated competitor research
            - Customer persona development
            - ICP identification
            """,
            3: """
            You are helping with Step 3: Stakeholder & Customer Interviews.
            
            Focus on gathering insights from stakeholders and customers. Ask about:
            - Key stakeholders
            - Interview insights
            - Customer feedback
            - Market research
            """,
            4: """
            You are helping with Step 4: Category & Competitive Positioning.
            
            Focus on competitive landscape and market positioning. Ask about:
            - Competitors
            - Market category
            - Differentiation
            - Competitive advantages
            """,
            5: """
            You are helping with Step 5: Feature/Benefit Translation ("Product Legos").
            
            Focus on translating features into benefits. Ask about:
            - Key features
            - Customer benefits
            - Value propositions
            - Product capabilities
            """,
            6: """
            You are helping with Step 6: Positioning Statement Creation.
            
            Focus on creating clear positioning statements. Ask about:
            - Target audience
            - Market category
            - Key benefit
            - Proof points
            """,
            7: """
            You are helping with Step 7: Positioning Workshop Facilitation.
            
            Focus on facilitating positioning workshops. Ask about:
            - Workshop participants
            - Key insights
            - Decisions made
            - Next steps
            """,
            8: """
            You are helping with Step 8: Messaging Framework & Brand Essence.
            
            Focus on developing messaging frameworks. Ask about:
            - Brand essence
            - Key messages
            - Tone and voice
            - Messaging hierarchy
            """,
            9: """
            You are helping with Step 9: Proof Points & Narrative.
            
            Focus on developing proof points and narratives. Ask about:
            - Supporting evidence
            - Customer stories
            - Case studies
            - Success metrics
            """,
            10: """
            You are helping with Step 10: Asset Generation & Application.
            
            Focus on creating marketing assets. Ask about:
            - Asset types needed
            - Content requirements
            - Distribution channels
            - Asset specifications
            """,
            11: """
            You are helping with Step 11: Asset Inventory & Message Testing.
            
            Focus on testing and optimizing messages. Ask about:
            - Testing methods
            - Performance metrics
            - Optimization opportunities
            - Final recommendations
            """
        }
        
        base_context = step_contexts.get(current_step, "You are helping with positioning and messaging.")
        
        # Add user context if available
        if user_responses:
            context_addition = f"\n\nUser Context:\n"
            for key, value in user_responses.items():
                context_addition += f"- {key}: {value}\n"
            base_context += context_addition
        
        return base_context
    
    async def validate_response(self, message: str, question: str, question_type: str, session_id: str) -> bool:
        """Validate a user response using LangGraph"""
        
        validation_prompt = f"""
        You are a validator for user responses in a PMM Assistant conversational interface.
        
        Question: {question}
        Question Type: {question_type}
        User Response: "{message}"
        
        Evaluate this response based on:
        1. Is it relevant to the question asked?
        2. Is it a meaningful, helpful response?
        3. Does it provide useful information?
        
        IMPORTANT: You must respond with EXACTLY one word:
        - "VALID" if the response is relevant, meaningful, and helpful
        - "INVALID" if the response is too generic, unhelpful, or meaningless
                
        Note: Company names, business descriptions, and clear answers are VALID even if brief.
        
        Response (VALID or INVALID only):
        """
        
        try:
            response = self.model.invoke([HumanMessage(content=validation_prompt)])
            
            # Parse the LLM response more carefully
            response_text = response.content.strip().upper()
            print(f"LLM validation response for '{message}': {response_text}")
            
            # Check for explicit VALID/INVALID
            if "VALID" in response_text and "INVALID" not in response_text:
                return True
            elif "INVALID" in response_text:
                return False
            else:
                # If LLM didn't follow instructions, default to invalid
                print(f"LLM returned unclear response: {response_text}. Defaulting to INVALID.")
                return False
                
        except Exception as e:
            print(f"Validation error: {e}")
            # If there's an error, default to invalid to be safe
            return False
    
    async def generate_pmm_plan(self, responses: Dict[str, Any], session_id: str) -> str:
        """Generate a personalized PMM plan based on Step 1 responses"""
        
        # Extract key information from responses
        company_name = responses.get("company_name", "Your company")
        company_description = responses.get("company_description", "your business")
        customer_description = responses.get("customer_description", "your customers")
        positioning_experience = responses.get("positioning_experience", "positioning")
        company_scope = responses.get("company_scope", "your company")
        product_name = responses.get("product_name", "")
        
        # Build context string
        context_info = f"""
        Company Information:
        - Company Name: {company_name}
        - Business Description: {company_description}
        - Customer Description: {customer_description}
        - Positioning Experience: {positioning_experience}
        - Company Scope: {company_scope}"""
        
        if product_name:
            context_info += f"\n        - Product Name: {product_name}"
        
        plan_prompt = f"""
        Based on the following information about {company_name}, generate a personalized PMM (Positioning & Messaging) plan that outlines how we'll help them through our 11-step workflow.
        
        {context_info}
        
        Create a brief, personalized plan that mentions all 11 steps of our PMM workflow:
        1. Context Gathering & Planning
        2. Customer Understanding & Persona Development
        3. Stakeholder & Customer Interviews
        4. Category & Competitive Positioning
        5. Feature/Benefit Translation ("Product Legos")
        6. Positioning Statement Creation
        7. Positioning Workshop Facilitation
        8. Messaging Framework & Brand Essence
        9. Proof Points & Narrative
        10. Asset Generation & Application
        11. Asset Inventory & Message Testing
        
        Make it conversational and specific to their business. Explain how each step will help them achieve their positioning and messaging goals.
        """
        
        try:
            response = self.model.invoke([HumanMessage(content=plan_prompt)])
            return response.content
        except Exception as e:
            print(f"Error generating PMM plan: {e}")
            # Build fallback plan
            scope_text = f"{company_scope.lower()}"
            if product_name:
                scope_text = f"{product_name} specifically"
            
            return f"""🎉 **Great! You've completed Step 1: Context Gathering & Planning**

Based on your responses about **{company_name}**, here's how I'll help you through our comprehensive PMM workflow:

**Your Personalized PMM Journey:**

**Step 2: Customer Understanding & Persona Development** - We'll dive deep into understanding your {customer_description} to create detailed customer personas.

**Step 3: Stakeholder & Customer Interviews** - We'll gather insights from key stakeholders and customers to inform your positioning.

**Step 4: Category & Competitive Positioning** - We'll analyze your competitive landscape and define your market category.

**Step 5: Feature/Benefit Translation ("Product Legos")** - We'll translate your {company_description} features into compelling customer benefits.

**Step 6: Positioning Statement Creation** - We'll craft clear, compelling positioning statements for **{company_name}**.

**Step 7: Positioning Workshop Facilitation** - We'll facilitate workshops to refine and validate your positioning.

**Step 8: Messaging Framework & Brand Essence** - We'll develop your core messaging framework and brand essence.

**Step 9: Proof Points & Narrative** - We'll create supporting evidence and compelling narratives.

**Step 10: Asset Generation & Application** - We'll develop marketing assets and content.

**Step 11: Asset Inventory & Message Testing** - We'll test and optimize your messaging for maximum impact.

Since you're doing this {positioning_experience.lower()} for {scope_text}, we'll tailor each step to your specific needs and goals."""

    async def process_message(self, message: str, session_id: str, current_step: int, previous_responses: Dict[str, Any]) -> str:
        """Process a user message through the workflow"""
        
        # Create initial state
        initial_state = WorkflowState(
            messages=[HumanMessage(content=message)],
            current_step=current_step,
            session_id=session_id,
            user_responses=previous_responses
        )
        
        # Run the workflow
        config = {"configurable": {"thread_id": session_id}}
        result = self.graph.invoke(initial_state, config)
        
        # Extract the final response
        if result.get("messages"):
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                return last_message.content
        
        return "I'm here to help with your positioning and messaging journey. How can I assist you?"
    
    async def conduct_competitor_research(self, session_id: str) -> str:
        """Conduct automated competitor research to identify ICP, buyers, influencers, and users"""
        
        # Get user responses from Step 1
        responses = self.get_session_responses(session_id)
        
        if not responses:
            return "No user responses found. Please complete Step 1 first."
        
        # Extract key information for research
        company_name = responses.get("company_name", "Unknown Company")
        company_description = responses.get("company_description", "Unknown business")
        customer_description = responses.get("customer_description", "Unknown customers")
        
        research_prompt = f"""
        You are a market research analyst conducting automated competitor research. Based on the following company information, research and identify their ICP, buyers, influencers, and users by analyzing competitor websites and market data.

        Company: {company_name}
        Business: {company_description}
        Current Customers: {customer_description}

        Conduct comprehensive research and provide detailed insights on:

        1. **Ideal Customer Profile (ICP)**:
           - Demographics and firmographics
           - Industry verticals and company sizes
           - Technology stack and preferences
           - Pain points and challenges

        2. **Buyer Personas**:
           - Decision makers vs end users
           - Roles and responsibilities
           - Buying journey stages
           - Decision criteria and influence factors

        3. **Key Influencers**:
           - Industry thought leaders
           - Internal champions
           - Community advocates
           - Media and analyst relationships

        4. **User Personas**:
           - End-user characteristics
           - Usage patterns and behaviors
           - Feature preferences
           - Success metrics

        Format your response as a comprehensive research report with clear sections and actionable insights.
        """
        
        try:
            response = self.model.invoke([HumanMessage(content=research_prompt)])
            return response.content
        except Exception as e:
            logger.error(f"Error conducting competitor research: {str(e)}")
            return f"Error conducting research: {str(e)}"
    
    def get_session_responses(self, session_id: str) -> Dict[str, Any]:
        """Get user responses for a session (placeholder - in production, this would query a database)"""
        # This is a placeholder method. In a real application, you would:
        # 1. Query your database for the session
        # 2. Return the stored responses
        # For now, return empty dict to avoid errors
        return {}
    
    async def conduct_competitor_research_with_responses(self, session_id: str, responses: Dict[str, Any]) -> str:
        """Conduct automated competitor research using provided user responses"""
        
        if not responses:
            return "No user responses found. Please complete Step 1 first."
        
        # Extract key information for research
        company_name = responses.get("company_name", "Unknown Company")
        company_description = responses.get("company_description", "Unknown business")
        customer_description = responses.get("customer_description", "Unknown customers")
        
        research_prompt = f"""
        You are a market research analyst conducting automated competitor research. Based on the following company information, research and identify their ICP, buyers, influencers, and users by analyzing competitor websites and market data.

        Company: {company_name}
        Business: {company_description}
        Current Customers: {customer_description}

        Conduct comprehensive research and provide detailed insights on:

        1. **Ideal Customer Profile (ICP)**:
           - Demographics and firmographics
           - Industry verticals and company sizes
           - Technology stack and preferences
           - Pain points and challenges

        2. **Buyer Personas**:
           - Decision makers vs end users
           - Roles and responsibilities
           - Buying journey stages
           - Decision criteria and influence factors

        3. **Key Influencers**:
           - Industry thought leaders
           - Internal champions
           - Community advocates
           - Media and analyst relationships

        4. **User Personas**:
           - End-user characteristics
           - Usage patterns and behaviors
           - Feature preferences
           - Success metrics

        Format your response as a comprehensive research report with clear sections and actionable insights.
        """
        
        try:
            response = self.model.invoke([HumanMessage(content=research_prompt)])
            return response.content
        except Exception as e:
            logger.error(f"Error conducting competitor research: {str(e)}")
            return f"Error conducting research: {str(e)}"
