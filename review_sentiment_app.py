import streamlit as st
import re
import string
import json
import nltk
import tiktoken
from nltk.corpus import stopwords
from langchain_fireworks import ChatFireworks
from langchain_core.messages import HumanMessage, SystemMessage
# import logging

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text)
    text = text.strip()
    text = text.lower()  
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


def ask_llm(review):
    import logging

    for noisy_logger in ["httpx", "fireworks", "urllib3"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    review = clean_text(review)

    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # logger = logging.getLogger(__name__)

    def count_tokens(text, model="accounts/fireworks/models/llama4-maverick-instruct-basic"):
        try:    
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))

    if not isinstance(review, str) or not review.strip():
        return "Empty input!"
    
    else:
        schema = {
            "type": "object",
            "properties": {
                "output": {
                    "type": "string",
                    "description": "The single review feedback or 'Empty input!'"
                }
            },
            "required": ["output"],
            "additionalProperties": False
        }

        model = ChatFireworks(
            model="accounts/fireworks/models/llama4-scout-instruct-basic",
            temperature=0.1,
            max_tokens=131072, 
            fireworks_api_key="fw_3ZTtMu8yiAV962ddVBRTHBAD",
            top_p=0.3, 
            top_k=3,
        )

        prompt = f"""
        You are a sentiment analysis system. Your task is to return the food review feedback based on a customer's review.
        Your response will be ONLY (Positve) or (Negative)

        Inputs:
        - Review: "{review}"
        
        Instructions:
        - Choose the single best matching feedback for the review.
        - Output format must be strictly: {{"output": "<exact feeback or Empty input!>" }}
        - No extra text, explanation, or formatting

        Examples:
        - Review: "very bad try" → {{ "output": "Negative" }}
        - Review: "what a wonderful place" → {{ "output": "Potive" }}

        predict feedback for this review: "{review}"
        """

        json_model = model.bind(response_format={"type": "json_object", "schema": schema})
        chat_history = [SystemMessage(content=prompt), HumanMessage(content=review)]
        
        system_tokens = count_tokens(prompt)
        human_tokens = count_tokens(review)
        total_input_tokens = system_tokens + human_tokens
        
        # logger.info(f"INPUT TOKENS: {total_input_tokens} (System: {system_tokens}, Human: {human_tokens})")
        # logger.info(f"System prompt: {prompt}")
        # logger.info(f"Human message: {review}")

        try:
            response = json_model.invoke(chat_history)
            content = response.content.strip()
            
            output_tokens = count_tokens(content)
            # logger.info(f"OUTPUT TOKENS: {output_tokens}")
            
            parsed = json.loads(content)
            feedback = parsed.get("output", "Empty input!")
            
            # logger.info(f"TOTAL TOKENS: {total_input_tokens + output_tokens} (Input: {total_input_tokens}, Output: {output_tokens})")
            # logger.info(f"Feedback: {feedback}")
            
            return feedback

        except Exception as e:
            # logger.error(f"ERROR: {str(e)}")
            return "Empty input!"


st.set_page_config(page_title="Review Sentiment Predictor", page_icon="🧠")

st.title("🧠 Food Review Sentiment Classifier")
st.markdown("Enter your feedback below and see if it's **Positive** or **Negative**.")

user_review = st.text_area("enter your review, please", placeholder="Type your review here...", height=150)

if st.button("Predict Sentiment"):
    if not user_review.strip():
        st.warning("Please enter a review before predicting.")
    else:
        with st.spinner("Analyzing review..."):
            feedback = ask_llm(user_review)
            if feedback == "Positive":
                st.success("✅ Sentiment: Positive")
            elif feedback == "Negative":
                st.error("❌ Sentiment: Negative")
            else:
                st.info("ℹ️ Result: Empty input or unclear")
