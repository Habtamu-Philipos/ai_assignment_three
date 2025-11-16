# ================================================
# TASK 3: NLP with spaCy (Amazon Reviews)
# ================================================

# Install required packages (run once in terminal or notebook)
# !pip install spacy textblob
# !python -m spacy download en_core_web_sm

# Import libraries
import spacy
from textblob import TextBlob

# Step 2: Load spaCy English model
nlp = spacy.load('en_core_web_sm')

# Sample Amazon reviews
reviews = [
    "I love my new Apple iPhone. The camera is amazing, and Apple has outdone themselves.",
    "The Samsung TV has great picture quality, but the sound is a bit low.",
    "Bought a Sony PlayStation, but it arrived damaged. Bad experience with Amazon delivery.",
    "This Canon camera is perfect for photography enthusiasts.",
    "The Bose headphones are comfortable and have excellent noise cancellation."
]

# Step 3 & 4: Process each review with NER and rule-based sentiment
print("NLP Analysis of Amazon Reviews:\n" + "="*60)
for idx, review in enumerate(reviews, 1):
    # Perform NER using spaCy
    doc = nlp(review)
    
    # Extract entities labeled ORG or PRODUCT
    entities = [(ent.text, ent.label_) for ent in doc.ents 
                if ent.label_ in ['ORG', 'PRODUCT']]
    
    # Compute sentiment using TextBlob
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    # Print formatted output
    print(f"Review {idx}:")
    print(f"   Text: {review}")
    print(f"   Entities: {entities}")
    print(f"   Sentiment: {sentiment} (polarity: {polarity:.3f})\n")