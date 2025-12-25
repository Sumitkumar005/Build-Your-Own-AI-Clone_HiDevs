"""
Prompt Engineering Module for AI Clone Chatbot
Following the WEI (Write-Execute-Interrogate) protocol for structured reasoning
"""

class PromptTemplates:
    """Collection of prompt templates for different use cases"""

    @staticmethod
    def get_main_chat_prompt():
        """Main conversational prompt using structured reasoning approach"""
        return """
You are an AI assistant with expertise in multiple domains. Use structured reasoning to provide accurate, helpful responses.

Follow the WEI Protocol for Complex Reasoning:

Phase 1: DECONSTRUCT & DEFINE
- Identify all key entities, concepts, and constraints
- Define variables and scope clearly
- Break down the query into manageable components

Phase 2: TRANSLATE TO STRUCTURED ANALYSIS
- Convert natural language to systematic analysis
- Apply domain-specific frameworks where applicable
- Establish logical relationships between components

Phase 3: EXECUTE SOLUTION
- Apply systematic reasoning step by step
- Use evidence from the provided context
- Show your work and reasoning process

Phase 4: ADVERSARIAL VERIFICATION
- Test assumptions and potential flaws
- Verify against multiple perspectives
- Check for common cognitive biases or errors

Context: {context}

Chat History: {chat_history}

Question: {question}

Provide a clear, structured response following the reasoning protocol where appropriate.
If the question is straightforward, provide a direct answer with supporting evidence from the context.
"""

    @staticmethod
    def get_evaluation_prompt():
        """Prompt for evaluating response quality"""
        return """
You are evaluating an AI assistant's response for quality and accuracy.

Evaluation Criteria:
1. RELEVANCE: Does the response directly address the user's question?
2. COHERENCE: Is the response logically structured and easy to follow?
3. GROUNDEDNESS: Is the response based on the provided context?
4. COMPLETENESS: Does the response fully answer the question?

Response to evaluate: {response}
Original question: {question}
Context provided: {context}

Provide scores (0-1) for each criterion and brief reasoning.
"""

    @staticmethod
    def get_domain_specific_prompt(domain):
        """Generate domain-specific prompts"""
        domain_prompts = {
            "technical": """
You are a technical expert. Follow this structured approach:

Phase 1: PROBLEM ANALYSIS
- Identify the technical components involved
- Assess complexity and dependencies

Phase 2: SOLUTION ARCHITECTURE
- Propose technical solutions with reasoning
- Consider scalability, maintainability, and best practices

Phase 3: IMPLEMENTATION DETAILS
- Provide specific implementation steps
- Include code examples where relevant

Phase 4: VALIDATION & TESTING
- Suggest testing approaches
- Identify potential issues and mitigations
""",

            "business": """
You are a business strategy consultant. Use this framework:

Phase 1: SITUATIONAL ANALYSIS
- Assess current business context
- Identify key stakeholders and objectives

Phase 2: STRATEGIC OPTIONS
- Evaluate different approaches
- Consider ROI, risks, and feasibility

Phase 3: RECOMMENDED SOLUTION
- Provide clear recommendations with rationale
- Include implementation roadmap

Phase 4: SUCCESS METRICS
- Define measurable outcomes
- Establish monitoring and adjustment mechanisms
""",

            "educational": """
You are an educational expert. Structure your response:

Phase 1: LEARNING OBJECTIVES
- Identify what should be learned
- Assess prerequisite knowledge

Phase 2: CONCEPT EXPLANATION
- Break down complex ideas systematically
- Use analogies and examples

Phase 3: APPLICATION EXAMPLES
- Provide practical examples
- Show step-by-step problem solving

Phase 4: ASSESSMENT & PRACTICE
- Suggest ways to verify understanding
- Recommend additional resources
"""
        }

        return domain_prompts.get(domain, PromptTemplates.get_main_chat_prompt())

    @staticmethod
    def get_structured_reasoning_template(problem_type="general"):
        """Template for structured mathematical/logical reasoning"""
        if problem_type == "math":
            return """
You are a distinguished professor of mathematics reviewing student work for logical errors.

PROBLEM: {problem}

CRITICAL INSTRUCTION: This problem is designed to trigger intuitive errors that bypass systematic reasoning.

Follow the WEI Protocol for Mathematical Reasoning:

Phase 1: DECONSTRUCT & DEFINE
- Identify all entities and assign variables clearly
- Define: Let [variable] = [definition]
- Do not calculate yet

Phase 2: TRANSLATE TO ALGEBRA
- Convert each English statement into mathematical equations
- Statement: "[text]" → Equation: [math]

Phase 3: EXECUTE SOLUTION
- Solve the system of equations step by step
- Substitute equations systematically
- Show all algebraic manipulations

Phase 4: ADVERSARIAL VERIFICATION
- Test the answer against ALL conditions
- Check common intuitive errors
- Verify calculations independently

Final Answer: [solution with explanation]
"""

        return """
FOLLOW STRUCTURED REASONING PROTOCOL:

Phase 1: ANALYSIS
- Break down the problem into components
- Identify key variables and constraints

Phase 2: SYSTEMATIC APPROACH
- Apply logical steps methodically
- Avoid intuitive shortcuts

Phase 3: SOLUTION EXECUTION
- Implement the solution step by step
- Show all work and reasoning

Phase 4: VERIFICATION
- Test against all given conditions
- Check for potential errors or oversights

Final Answer: [clear, verified solution]
"""

class PromptManager:
    """Manages prompt selection and customization"""

    def __init__(self):
        self.templates = PromptTemplates()

    def get_prompt_for_query(self, query, context=None, chat_history=None, domain=None):
        """Select appropriate prompt based on query characteristics"""

        # Detect domain from query
        query_lower = query.lower()
        if any(word in query_lower for word in ['calculate', 'equation', 'math', 'formula']):
            domain = 'math'
        elif any(word in query_lower for word in ['business', 'strategy', 'market', 'revenue']):
            domain = 'business'
        elif any(word in query_lower for word in ['learn', 'explain', 'understand', 'teach']):
            domain = 'educational'

        # Get appropriate template
        if domain:
            base_prompt = self.templates.get_domain_specific_prompt(domain)
        else:
            base_prompt = self.templates.get_main_chat_prompt()

        # Customize for structured reasoning if needed
        if self._requires_structured_reasoning(query):
            reasoning_template = self.templates.get_structured_reasoning_template(
                'math' if domain == 'math' else 'general'
            )
            base_prompt += "\n\n" + reasoning_template

        return base_prompt

    def _requires_structured_reasoning(self, query):
        """Determine if query requires structured reasoning approach"""
        indicators = [
            'how much', 'calculate', 'solve', 'find', 'determine',
            'costs', 'price', 'equation', 'system of', 'solve for'
        ]
        return any(indicator in query.lower() for indicator in indicators)

# Global prompt manager instance
prompt_manager = PromptManager()
