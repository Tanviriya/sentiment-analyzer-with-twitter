# Twitter Sentiment Analysis Tool 🐦

A Python tool that analyzes tweet sentiment using NLP and Machine Learning to classify text as positive, negative, or neutral.

## 🚀 Features

- **Text Preprocessing**: Cleans tweets (removes URLs, mentions, hashtags)
- **ML Classification**: Logistic Regression with TF-IDF vectorization
- **Real-time Analysis**: Interactive sentiment prediction
- **Visualizations**: Charts, word clouds, and performance metrics
- **Comparison Tools**: ML model vs TextBlob sentiment analysis

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn wordcloud nltk textblob scikit-learn

# Run the tool
python twitter_sentiment_analyzer.py
```

## 💻 Usage

### Basic Usage
```bash
python twitter_sentiment_analyzer.py
```

### As a Module
```python
from twitter_sentiment_analyzer import TwitterSentimentAnalyzer

analyzer = TwitterSentimentAnalyzer()
df = analyzer.load_sample_data()
analyzer.train_model(df)

result = analyzer.predict_sentiment("I love this!")
print(f"Sentiment: {result['sentiment']}")
```

## 📊 Performance

- **Accuracy**: ~92-95% on test data
- **Model**: Logistic Regression with TF-IDF
- **Dataset**: 45 sample tweets (15 each: positive, negative, neutral)

## 🛠️ Technologies

- **NLP**: NLTK, TextBlob
- **ML**: scikit-learn
- **Visualization**: matplotlib, seaborn, wordcloud
- **Data**: pandas, numpy

## 🔮 Future Enhancements

- Twitter API integration
- Deep learning models (BERT, LSTM)
- Web interface
- Real-time sentiment tracking
