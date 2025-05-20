# Zomato Review Sentiment Predictor

- This project analyzes Zomato restaurant data to predict user feedback (positive or negative) for reviews using a Large Language Model (LLM).
  
- The dataset, containing 51,717 entries, was preprocessed to handle missing values and standardize features like ratings, votes, and cuisines. Key features include clustering restaurants by service attributes and a Streamlit app for interactive sentiment prediction.

- The app leverages NLP techniques (e.g., text cleaning, stopwords removal) and the langchain_fireworks LLM to classify reviews in real-time.

- You can find the the preprocessing steps, analysis and LLM creation in the file **[code.ipynb]** file, and about the streamlit page itself in the file **[review_sentiment_app.py]**

- Check out the live demo [https://zomato-reviews-predictor.streamlit.app/] to test your restaurant reviews!

# Tech Stack: 
- Python, Pandas, Scikit-learn, Streamlit, NLTK, LangChain, Fireworks LLM


# How to Run:
- Clone the repo: git clone <repo-link>

- Install dependencies: pip install -r **requirements.txt**

- Run the app: streamlit run **review_sentiment_app.py**
