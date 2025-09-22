// PMM Assistant Frontend JavaScript

class PMMAssistant {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.currentStep = 1;
        this.currentQuestionIndex = 0;
        this.questions = [];
        this.responses = {};
        this.isInChatMode = false;
        
        this.init();
    }
    
    generateSessionId() {
        return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }
    
    async init() {
        await this.startWorkflow();
        this.startChatMode();
        this.setupEventListeners();
        this.updateProgress();
    }
    
    async startWorkflow() {
        try {
            const response = await fetch('/api/start-workflow', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ session_id: this.sessionId })
            });
            
            const data = await response.json();
            console.log('Workflow started:', data);
        } catch (error) {
            console.error('Error starting workflow:', error);
            this.showError('Failed to start workflow. Please refresh the page.');
        }
    }
    
    
    
    
    async startChatMode() {
        this.isInChatMode = true;
        document.getElementById('question-section').classList.add('d-none');
        document.getElementById('completion-section').classList.add('d-none');
        document.getElementById('chat-section').classList.remove('d-none');
        
        // Add welcome message
        this.addChatMessage('assistant', 'Welcome to the PMM Assistant! I\'ll guide you through an 11-step positioning and messaging workflow. Let\'s start with Step 1: Context Gathering & Planning.');
        
        // Start with first question
        this.currentQuestionIndex = 0;
        this.addChatMessage('assistant', 'What is your company name?');
        this.questions = [
            {
                id: "company_name",
                question: "What is your company name?",
                type: "text",
                required: true
            },
            {
                id: "company_description", 
                question: "Please provide a brief description of what your company does.",
                type: "textarea",
                required: true
            },
            {
                id: "customer_description",
                question: "Please provide a brief description of who your customers are.",
                type: "textarea", 
                required: true
            },
            {
                id: "positioning_experience",
                question: "Are you doing the positioning and messaging exercise for the first time, or are you repositioning an existing brand?",
                type: "select",
                options: ["First time", "Repositioning existing brand"],
                required: true
            },
            {
                id: "company_scope",
                question: "Are you doing this exercise for the whole company or a specific segment/product?",
                type: "select",
                options: ["Whole company", "Specific segment/product"],
                required: true
            },
            {
                id: "product_name",
                question: "What is the name of the specific product you're doing this exercise for?",
                type: "text",
                required: false,
                conditional: true,
                depends_on: "company_scope",
                depends_value: "Specific segment/product"
            }
        ];
    }
    
    async sendChatMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message) return;
        
        // Add user message to chat
        this.addChatMessage('user', message);
        input.value = '';
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            // If we're still in Step 1 questions, handle them
            if (this.currentQuestionIndex < this.questions.length) {
                await this.handleStep1Question(message);
            } else {
                // Use LangGraph workflow for subsequent steps
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        session_id: this.sessionId,
                        message: message
                    })
                });
                
                const data = await response.json();
                
                // Remove typing indicator
                this.hideTypingIndicator();
                
                // Add assistant response
                this.addChatMessage('assistant', data.response);
            }
            
        } catch (error) {
            console.error('Error sending chat message:', error);
            this.hideTypingIndicator();
            this.addChatMessage('assistant', 'I apologize, but I encountered an error. Please try again.');
        }
    }
    
    async handleStep1Question(message) {
        const currentQuestion = this.questions[this.currentQuestionIndex];
        
        // Validate response using LangGraph
        const isValid = await this.validateResponse(message, currentQuestion);
        
        if (!isValid) {
            // Remove typing indicator
            this.hideTypingIndicator();
            // Ask for clarification and repeat the same question
            this.addChatMessage('assistant', 'I need a more detailed response. Could you please provide more specific information?');
            this.addChatMessage('assistant', currentQuestion.question);
            return;
        }
        
        // Store the response
        this.responses[currentQuestion.id] = message;
        
        // Move to next question
        this.currentQuestionIndex++;
        
        // Remove typing indicator
        this.hideTypingIndicator();
        
        if (this.currentQuestionIndex < this.questions.length) {
            // Check if we need to ask the next question
            const nextQuestion = this.questions[this.currentQuestionIndex];
            
            // Check if this is a conditional question
            if (nextQuestion.conditional && nextQuestion.depends_on) {
                const dependsResponse = this.responses[nextQuestion.depends_on];
                if (dependsResponse !== nextQuestion.depends_value) {
                    // Skip this conditional question
                    this.currentQuestionIndex++;
                    // Check if there are more questions after skipping
                    if (this.currentQuestionIndex < this.questions.length) {
                        this.handleStep1Question(""); // Recursive call to continue
                    } else {
                        // No more questions, complete Step 1
                        await this.saveStep1Responses();
                        await this.generatePMMPlan();
                        this.updateStepStatus(1, 'completed');
                        this.updateProgress();
                    }
                    return;
                }
            }
            
            this.addChatMessage('assistant', nextQuestion.question);
        } else {
            // Step 1 complete, save responses and generate personalized plan
            await this.saveStep1Responses();
            await this.generatePMMPlan();
            this.updateStepStatus(1, 'completed');
            this.updateProgress();
        }
    }
    
    async validateResponse(message, question) {
        try {
            const response = await fetch('/api/validate-response', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    message: message,
                    question: question.question,
                    question_type: question.type
                })
            });
            
            const data = await response.json();
            return data.is_valid;
        } catch (error) {
            console.error('Error validating response:', error);
            // If validation fails, default to invalid to be safe
            return false;
        }
    }
    
    async saveStep1Responses() {
        try {
            // Save all Step 1 responses
            for (const [questionId, response] of Object.entries(this.responses)) {
                const responseData = {
                    step: 1,
                    question_id: questionId,
                    response: response,
                    session_id: this.sessionId
                };
                
                await fetch('/api/submit-response', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(responseData)
                });
            }
        } catch (error) {
            console.error('Error saving Step 1 responses:', error);
        }
    }
    
    async generatePMMPlan() {
        // Show loading message
        this.addChatMessage('assistant', '🎯 **Generating your personalized positioning and messaging strategy...**\n\nPlease wait while I create a customized roadmap based on your responses.');
        
        try {
            const response = await fetch('/api/generate-plan', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    responses: this.responses
                })
            });
            
            const data = await response.json();
            this.addChatMessage('assistant', data.plan);
            // Transition to Step 2 - Customer Understanding & Persona Development
            setTimeout(() => {
                this.currentStep = 2;
                this.updateStepStatus(2, 'in_progress');
                this.updateProgress();
                
                // Start Step 2 with document upload question
                this.addChatMessage('assistant', 'Excellent! Now let\'s move to **Step 2: Customer Understanding & Persona Development**. This is where we\'ll dive deep into understanding your target customers.');
                this.addChatMessage('assistant', 'Do you have existing user personas or customer research documents? If yes, please upload them. If not, I can help you create a research plan.');
                this.showDocumentUpload();
            }, 2000);
        } catch (error) {
            console.error('Error generating PMM plan:', error);
            this.addChatMessage('assistant', 'Great! You\'ve completed Step 1: Context Gathering & Planning. Now let\'s dive deeper into your positioning and messaging. What would you like to explore first?');
        }
    }
    
    addChatMessage(sender, message) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageElement = document.createElement('div');
        messageElement.className = `chat-message ${sender}`;
        
        // Format message with basic markdown-like formatting
        const formattedMessage = this.formatMessage(message);
        messageElement.innerHTML = formattedMessage;
        
        messagesContainer.appendChild(messageElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    formatMessage(message) {
        // Convert basic markdown-like formatting to HTML
        return message
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
            .replace(/\*(.*?)\*/g, '<em>$1</em>') // Italic
            .replace(/\n\n/g, '<br><br>') // Double line breaks
            .replace(/\n/g, '<br>'); // Single line breaks
    }
    
    showTypingIndicator() {
        const messagesContainer = document.getElementById('chat-messages');
        const typingElement = document.createElement('div');
        typingElement.className = 'chat-message assistant typing-indicator';
        typingElement.innerHTML = '<em>Assistant is typing...</em>';
        typingElement.id = 'typing-indicator';
        
        messagesContainer.appendChild(typingElement);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    updateProgress() {
        const completedSteps = document.querySelectorAll('.step-item.completed').length;
        const progressPercent = (completedSteps / 11) * 100;
        
        document.getElementById('overall-progress').style.width = `${progressPercent}%`;
        document.getElementById('current-step-number').textContent = this.currentStep;
    }
    
    updateStepStatus(stepNumber, status) {
        const stepElement = document.querySelector(`[data-step="${stepNumber}"]`);
        if (stepElement) {
            stepElement.className = `step-item ${status}`;
            
            const statusIcon = stepElement.querySelector('.status-icon');
            const statusText = stepElement.querySelector('.status-text');
            
            if (status === 'completed') {
                statusIcon.textContent = '✅';
                statusText.textContent = 'Completed';
            } else if (status === 'active') {
                statusIcon.textContent = '🔄';
                statusText.textContent = 'In Progress';
            } else {
                statusIcon.textContent = '⏳';
                statusText.textContent = 'Pending';
            }
        }
    }
    
    showLoading() {
        const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
        modal.show();
    }
    
    hideLoading() {
        const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
        if (modal) modal.hide();
    }
    
    showError(message) {
        // Simple error display - could be enhanced with a proper modal
        alert(message);
    }
    
    showDocumentUpload() {
        // Create file upload interface
        const uploadHTML = `
            <div class="document-upload-section mt-3">
                <div class="card">
                    <div class="card-body">
                        <h6>📄 Upload Persona Document</h6>
                        <input type="file" id="persona-file" class="form-control mb-2" accept=".pdf,.doc,.docx,.txt">
                        <button class="btn btn-primary btn-sm" onclick="window.app.uploadPersonaDocument()">Upload Document</button>
                        <button class="btn btn-outline-secondary btn-sm ms-2" onclick="window.app.sayNoPersonas()">I don't have personas</button>
                    </div>
                </div>
            </div>
        `;
        
        // Add to chat
        const chatMessages = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant-message';
        messageDiv.innerHTML = uploadHTML;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    async uploadPersonaDocument() {
        const fileInput = document.getElementById('persona-file');
        const file = fileInput.files[0];
        
        if (!file) {
            this.addChatMessage('assistant', 'Please select a file to upload.');
            return;
        }
        
        this.addChatMessage('user', `Uploaded: ${file.name}`);
        
        try {
            const response = await fetch('/api/upload-persona-document', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    filename: file.name
                })
            });
            
            const data = await response.json();
            this.addChatMessage('assistant', '✅ Document uploaded successfully! Now let me conduct automated competitor research to identify your ICP, buyers, influencers, and users.');
            
            // Remove upload interface
            const uploadSection = document.querySelector('.document-upload-section');
            if (uploadSection) uploadSection.remove();
            
            // Conduct research
            await this.conductCompetitorResearch();
            
        } catch (error) {
            console.error('Error uploading document:', error);
            this.addChatMessage('assistant', 'Sorry, there was an error uploading your document. Please try again.');
        }
    }
    
    sayNoPersonas() {
        this.addChatMessage('user', "I don't have existing personas");
        this.showCustomerResearchTodo();
        
        // Remove upload interface
        const uploadSection = document.querySelector('.document-upload-section');
        if (uploadSection) uploadSection.remove();
    }
    
    async showCustomerResearchTodo() {
        try {
            const response = await fetch('/api/get-customer-research-todo', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await response.json();
            this.addChatMessage('assistant', data.todo_list);
            
            // Ask if they want to proceed with automated research
            setTimeout(() => {
                this.addChatMessage('assistant', 'Would you like me to conduct automated competitor research to help identify your ICP, buyers, influencers, and users? I can analyze competitor websites and market data to provide insights.');
            }, 1000);
            
        } catch (error) {
            console.error('Error getting todo list:', error);
            this.addChatMessage('assistant', 'Sorry, there was an error generating the research plan. Please try again.');
        }
    }
    
    async conductCompetitorResearch() {
        this.addChatMessage('assistant', '🔍 **Conducting automated competitor research...**\n\nI\'m analyzing competitor websites, industry reports, and market data to identify your ICP, buyers, influencers, and users. This may take a moment.');
        
        try {
            const response = await fetch('/api/conduct-competitor-research', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.sessionId
                })
            });
            
            const data = await response.json();
            
            // Add research results
            this.addChatMessage('assistant', '## 📊 **Competitor Research Results**\n\n' + data.research_results);
            
            // Complete Step 2
            this.updateStepStatus(2, 'completed');
            this.updateProgress();
            
            // Move to Step 3
            setTimeout(() => {
                this.addChatMessage('assistant', 'Excellent! Step 2 is complete. Now let\'s move to Step 3: Stakeholder & Customer Interviews. What would you like to explore first?');
                this.currentStep = 3;
            }, 2000);
            
        } catch (error) {
            console.error('Error conducting research:', error);
            this.addChatMessage('assistant', 'Sorry, there was an error conducting the research. Please try again.');
        }
    }
    
    setupEventListeners() {
        // Chat input
        document.getElementById('chat-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendChatMessage();
            }
        });
        
        // Send chat button
        document.getElementById('send-chat-btn').addEventListener('click', () => {
            this.sendChatMessage();
        });
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.app = new PMMAssistant();
});
