import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("NLTK data downloaded successfully!")

class TwitterSentimentAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.pipeline = None
        
    def preprocess_text(self, text):
        """
        Preprocess tweet text for sentiment analysis
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def load_sample_data(self):
        """
        Create sample Twitter data for demonstration
        """
        sample_tweets = [
            # Positive tweets
            "I love this new movie! It's absolutely amazing! 😍",
            "Just had the best coffee ever! Great start to the day ☕",
            "Excited for the weekend! Going to have so much fun 🎉",
            "Beautiful sunset today. Nature is incredible 🌅",
            "Happy birthday to my best friend! Love you so much 🎂",
            "Feeling grateful for my family and friends today ❤️",
            "Just finished reading an amazing book! Highly recommend 📚",
            "Perfect weather for a picnic today! 🌞",
            "Congratulations to the team on their victory! 🏆",
            "Going to the gym feels great! Love staying active 💪",
            "Amazing concert last night! The band was incredible 🎵",
            "So proud of my daughter's achievements! She's amazing 👏",
            "This vacation is exactly what I needed! Feeling refreshed 🏖️",
            "Great news! Got the job I wanted! Dreams do come true ✨",
            "Love my new apartment! Finally feels like home 🏠",
            
            # Negative tweets
            "This weather is terrible. I hate rainy days 😞",
            "Traffic is so bad today. Really frustrating 😤",
            "This food tastes awful. Not ordering from here again 🤢",
            "Work is so stressful. Need a vacation badly 😫",
            "Stuck in a boring meeting. When will this end? 😴",
            "My phone battery died again. So annoying! 🔋",
            "Lost my keys again. I'm so clumsy 🤦",
            "This restaurant has terrible service. Very disappointed 😠",
            "Can't believe it's Monday already. Weekend went by so fast 😢",
            "Feeling sick today. This flu is really getting to me 🤒",
            "Another delay on the train. This commute is horrible 🚂",
            "Failed my driving test again. So frustrated with myself 😤",
            "Terrible customer service experience. Never shopping here again 😡",
            "My laptop crashed and I lost all my work. This is a disaster 💻",
            "Dealing with a difficult client today. So exhausting 😩",
            
            # Neutral tweets
            "This movie was okay. Nothing special but not bad either",
            "The weather is cloudy today. Might rain later",
            "Had lunch at a new restaurant downtown. It was decent",
            "Meeting scheduled for 3 PM. Need to prepare presentation",
            "Just finished grocery shopping. Got everything I needed",
            "Watching the news right now. Lots happening in the world",
            "Taking the bus to work today. Train was cancelled",
            "Reading a book about history. Learning interesting facts",
            "Going to bed early tonight. Have an early meeting tomorrow",
            "The store was crowded but service was average",
            "Updated my resume today. Looking for new opportunities",
            "Attended a webinar on digital marketing. Some useful tips",
            "The movie starts at 7 PM. Will grab dinner before that",
            "Working from home today. Internet connection is stable",
            "Picked up my dry cleaning. Everything looks clean"
        ]
        
        # Labels: 1=positive, 0=negative, 2=neutral
        sample_labels = [1]*15 + [0]*15 + [2]*15
        
        return pd.DataFrame({
            'text': sample_tweets,
            'sentiment': sample_labels
        })
    
    def textblob_sentiment(self, text):
        """
        Get sentiment using TextBlob
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def train_model(self, df):
        """
        Train sentiment analysis model
        """
        # Preprocess texts
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Prepare data
        X = df['processed_text']
        y = df['sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Create pipeline with TF-IDF and Logistic Regression
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        # Train model
        self.pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        
        # Print evaluation metrics
        print("Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['negative', 'positive', 'neutral']))
        
        return X_test, y_test, y_pred
    
    def predict_sentiment(self, text):
        """
        Predict sentiment for a single text
        """
        if self.pipeline is None:
            return "Model not trained yet!"
        
        processed_text = self.preprocess_text(text)
        prediction = self.pipeline.predict([processed_text])[0]
        confidence = self.pipeline.predict_proba([processed_text]).max()
        
        sentiment_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
        
        return {
            'text': text,
            'sentiment': sentiment_map[prediction],
            'confidence': confidence,
            'textblob_sentiment': self.textblob_sentiment(text)
        }
    
    def analyze_multiple_tweets(self, tweets):
        """
        Analyze sentiment for multiple tweets
        """
        results = []
        for tweet in tweets:
            result = self.predict_sentiment(tweet)
            results.append(result)
        
        return pd.DataFrame(results)

# Main execution
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TwitterSentimentAnalyzer()
    
    print("🐦 Twitter Sentiment Analysis Tool 🐦")
    print("="*50)
    
    # Load sample data
    print("\n1. Loading sample Twitter data...")
    df = analyzer.load_sample_data()
    print(f"Loaded {len(df)} sample tweets")
    
    # Train model
    print("\n2. Training sentiment analysis model...")
    X_test, y_test, y_pred = analyzer.train_model(df)
    
    # Test with new tweets
    print("\n3. Testing with new tweets...")
    test_tweets = [
        "I'm so excited about this new project! 🚀",
        "This is the worst day ever. Everything is going wrong 😭",
        "The weather is nice today. Perfect for a walk.",
        "I love spending time with my family ❤️",
        "This traffic is making me late for work 😤"
    ]
    
    results = analyzer.analyze_multiple_tweets(test_tweets)
    print("\nSentiment Analysis Results:")
    print(results[['text', 'sentiment', 'confidence']].to_string(index=False))
    
    # Interactive prediction
    print("\n4. Interactive Prediction Mode")
    print("Enter tweets to analyze (type 'quit' to exit):")
    
    while True:
        user_input = input("\nEnter a tweet: ")
        if user_input.lower() == 'quit':
            break
        
        result = analyzer.predict_sentiment(user_input)
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"TextBlob Sentiment: {result['textblob_sentiment']}")
    
    print("\n✅ Sentiment analysis complete!")
    print("\nNext steps:")
    print("- Connect to Twitter API for real-time data")
    print("- Expand training data with larger datasets")
    print("- Implement deep learning models (LSTM, BERT)")
    print("- Deploy as web application or API")
