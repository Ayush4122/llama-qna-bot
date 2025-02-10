import streamlit as st
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import random

class AdvancedFitnessChatbot:
    def __init__(self):
        # Initialize Llama 2 model
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        self.llm = LlamaCpp(
            model_path=""~/Models/llama-2-7b-chat.Q4_0.gguf"",  # Update with your model path
            temperature=0.7,
            max_tokens=2000,
            top_p=1,
            callback_manager=callback_manager,
            verbose=True,
        )

        # Enhanced knowledge base
        self.knowledge_base = self.load_comprehensive_knowledge()

        # Create prompt templates
        self.qa_template = PromptTemplate(
            input_variables=["question"],
            template="""You are a knowledgeable fitness expert. Please provide a detailed, 
            accurate answer to the following fitness-related question:
            
            Question: {question}
            
            Answer:"""
        )
        
        # Create LangChain chain
        self.qa_chain = LLMChain(llm=self.llm, prompt=self.qa_template)
        
        # Response templates for common interactions
        self.greeting_patterns = [
            "Hi there! I'm your fitness expert AI. How can I help you today?",
            "Hello! Ready to discuss fitness, nutrition, or exercise?",
            "Hey! Welcome to your personal fitness assistant.",
            "Greetings! What fitness goals can I help you with today?",
            "Welcome! Let's work together on your fitness journey."
        ]
        
        self.miscellaneous_responses = {
            "thanks": [
                "You're welcome! Always happy to help.",
                "Glad I could assist you.",
                "My pleasure! Fitness is my passion.",
                "Anytime! Keep up the great work!"
            ],
            "bye": [
                "Stay fit and healthy! Goodbye.",
                "Take care and keep moving!",
                "Wishing you success in your fitness journey!",
                "Keep crushing your fitness goals! Goodbye!"
            ]
        }


    def load_comprehensive_knowledge(self):
        """Load an extensive knowledge base covering multiple fitness domains"""
        return {
            "general_fitness": {
                "workout_types": [
                    "Strength training builds muscle and increases metabolism.",
                    "Cardiovascular exercise improves heart health and endurance.",
                    "Flexibility training prevents injuries and improves mobility.",
                    "High-Intensity Interval Training (HIIT) burns fat efficiently.",
                    "Bodyweight exercises can be done anywhere without equipment.",
                    "Olympic lifting develops explosive power and athletic performance.",
                    "Kettlebell training provides full-body workouts and functional strength."
                ],
                "fitness_goals": [
                    "Weight loss requires calorie deficit and consistent exercise.",
                    "Muscle gain needs progressive overload and proper nutrition.",
                    "Endurance training involves gradually increasing workout intensity.",
                    "Body recomposition combines fat loss and muscle gain strategies.",
                    "Power development requires explosive movements and proper technique.",
                    "Mobility improvement needs consistent stretching and proper form."
                ]
            },
            "nutrition": {
                "diet_principles": [
                    "Balanced macronutrients are crucial for optimal performance.",
                    "Protein intake supports muscle recovery and growth.",
                    "Hydration is key for metabolism and overall health.",
                    "Timing of meals impacts workout performance and recovery.",
                    "Micronutrients play vital roles in energy production and recovery.",
                    "Pre-workout nutrition should focus on easily digestible carbs."
                ],
                "meal_planning": [
                    "Prepare meals in advance to maintain consistent nutrition.",
                    "Include variety to ensure comprehensive nutrient intake.",
                    "Adjust calorie intake based on activity level and goals.",
                    "Time protein intake around workouts for optimal recovery.",
                    "Consider supplements to fill nutritional gaps."
                ]
            },
            "exercise_science": {
                "muscle_groups": [
                    "Compound exercises engage multiple muscle groups simultaneously.",
                    "Isolation exercises target specific muscle development.",
                    "Rest and recovery are essential for muscle growth.",
                    "Different rep ranges target various muscle fiber types.",
                    "Muscle tension time affects hypertrophy response."
                ],
                "injury_prevention": [
                    "Proper warm-up reduces injury risk.",
                    "Maintain correct form during all exercises.",
                    "Listen to your body and avoid overtraining.",
                    "Progressive overload should be implemented gradually.",
                    "Recovery techniques help prevent overuse injuries."
                ],
                "advanced_techniques": [
                    "Time under tension manipulates muscle growth stimulus.",
                    "Drop sets can help break through plateaus.",
                    "Super sets improve workout efficiency.",
                    "Periodization optimizes long-term progress.",
                    "Mind-muscle connection enhances exercise effectiveness."
                ]
            }
        }


    def classify_intent(self, query):
        """Simple intent classification based on keywords"""
        query = query.lower()
        
        if any(word in query for word in ['hi', 'hello', 'hey']):
            return 'greeting'
        elif any(word in query for word in ['thank', 'thanks']):
            return 'thanks'
        elif any(word in query for word in ['bye', 'goodbye']):
            return 'goodbye'
        
        # Use Llama 2 for more complex intent classification
        intent_prompt = PromptTemplate(
            input_variables=["query"],
            template="""Classify the following fitness-related query into one of these categories:
            - exercise
            - nutrition
            - recovery
            - technique
            - general
            
            Query: {query}
            Category:"""
        )
        
        intent_chain = LLMChain(llm=self.llm, prompt=intent_prompt)
        return intent_chain.run(query=query).strip().lower()

    def generate_response(self, query):
        """Generate response using Llama 2"""
        intent = self.classify_intent(query)
        
        # Handle basic intents
        if intent == 'greeting':
            return random.choice(self.greeting_patterns)
        elif intent == 'thanks':
            return random.choice(self.miscellaneous_responses['thanks'])
        elif intent == 'goodbye':
            return random.choice(self.miscellaneous_responses['bye'])
        
        # Generate response using Llama 2
        try:
            response = self.qa_chain.run(question=query)
            return response.strip()
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response. Please try rephrasing your question."

def main():
    st.title("🏋️ Advanced Fitness Expert AI")
    st.sidebar.info("Ask anything about fitness, nutrition, or exercise!")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = AdvancedFitnessChatbot()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("What fitness advice do you need?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display bot response
        with st.chat_message("assistant"):
            response = st.session_state.chatbot.generate_response(prompt)
            st.markdown(response)
        
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
