# === STEP 1: IMPORTS & NLTK SETUP ===
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

# === STEP 2: LOAD TEXT DATA ===
data = {
    "text": [
        "Natural Language Processing (NLP) plays a critical role in AI.",
        "NLP finds applications across many industries including healthcare, finance, and education.",
        "Text preprocessing is an important step in NLP pipelines."
    ]
}
df = pd.DataFrame(data)

# === STEP 3: DATA PREPROCESSING CHECKS ===
print("Initial Info:\n", df.info())
print("\nNull Values:\n", df.isnull().sum())
print("\nDuplicates:\n", df.duplicated().sum())

# === STEP 4: TOKENIZATION + CLEANING ===
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]  # Remove punctuation/numbers
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

df["tokens"] = df["text"].apply(clean_text)

# === STEP 5: STEMMING + LEMMATIZATION ===
df["stemmed"] = df["tokens"].apply(lambda words: [stemmer.stem(w) for w in words])
df["lemmatized"] = df["tokens"].apply(lambda words: [lemmatizer.lemmatize(w) for w in words])

# === STEP 6: POS TAGGING ===
df["pos_tags"] = df["tokens"].apply(pos_tag)

# === STEP 7: FLATTEN FOR VISUALIZATION ===
all_words = [word for sublist in df["lemmatized"] for word in sublist]
all_text = " ".join(all_words)

# === STEP 8: WORD CLOUD ===
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Cleaned Text")
plt.show()

# === STEP 9: TF-IDF (MANUAL) ===
# Build vocabulary
vocab = sorted(set(all_words))
wordDicts = []
for tokens in df["lemmatized"]:
    wordDict = dict.fromkeys(vocab, 0)
    for word in tokens:
        wordDict[word] += 1
    wordDicts.append(wordDict)

# TF Function
def computeTF(wordDict, doc):
    tf = {}
    doc_len = len(doc)
    for word, count in wordDict.items():
        tf[word] = count / doc_len
    return tf

# IDF Function
def computeIDF(docs):
    idf = dict.fromkeys(docs[0].keys(), 0)
    N = len(docs)
    for word in idf:
        idf[word] = math.log10(N / (1 + sum([doc[word] > 0 for doc in docs])))
    return idf

# Compute TF and IDF
tf_list = [computeTF(wordDicts[i], df["lemmatized"][i]) for i in range(len(df))]
idf = computeIDF(wordDicts)

# TF-IDF
tfidf_list = []
for tf in tf_list:
    tfidf = {word: tf[word] * idf[word] for word in vocab}
    tfidf_list.append(tfidf)

# Show TF-IDF Matrix
tfidf_df = pd.DataFrame(tfidf_list)
tfidf_df.index = ["Doc1", "Doc2", "Doc3"]
print("\nTF-IDF Matrix:\n", tfidf_df)

# === STEP 10: CORRELATION HEATMAP (OPTIONAL for word occurrence) ===
plt.figure(figsize=(10, 6))
sns.heatmap(tfidf_df.corr(), annot=True, cmap="YlGnBu")
plt.title("TF-IDF Word Correlation")
plt.show()
