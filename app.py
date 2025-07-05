from flask import Flask, render_template, request, redirect, url_for
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

app = Flask(__name__)

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

def predict_sentiment(text):
    # Get sentiment scores
    sentiment_score = sid.polarity_scores(text)
    
    # Analyze the text for negative keywords
    negative_keywords = ['bad', 'worst', 'terrible', 'horrible', 'awful', 'poor', 
                         'disappointed', 'waste', 'useless', 'broken', 'defective',
                         'not good', 'not worth', 'problem', 'issue', 'complaint']
    
    # Check for negative keywords in original text (case insensitive)
    text_lower = text.lower()
    has_negative_keywords = any(keyword in text_lower for keyword in negative_keywords)
    
    # Determine sentiment based on compound score and keywords
    if sentiment_score['compound'] <= -0.05 or has_negative_keywords:
        return 'Negative', sentiment_score
    else:
        return 'Positive', sentiment_score

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/analyze')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        sentiment, scores = predict_sentiment(text)
        
        # Calculate percentage for visualization
        if sentiment == 'Positive':
            percentage = int((scores['compound'] + 1) * 50)  # Convert from [-1,1] to [0,100]
        else:
            percentage = int((1 - scores['compound']) * 50)  # Convert from [-1,1] to [0,100]
            
        percentage = max(min(percentage, 100), 0)  # Ensure it's between 0 and 100
        
        return render_template('result.html', 
                              text=text, 
                              sentiment=sentiment, 
                              percentage=percentage,
                              pos_score=round(scores['pos']*100),
                              neg_score=round(scores['neg']*100),
                              neu_score=round(scores['neu']*100))

if __name__ == '__main__':
    print("Starting Sentiment Analysis application...")
    print("Access the application at http://127.0.0.1:5000/ or http://localhost:5000/")
    app.run(debug=True, host='0.0.0.0', port=5000)
