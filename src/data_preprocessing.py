import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# from nltk.stem import LancasterStemmer,WordNetLemmatizer
# from nltk.stem import WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer
# from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup


def preprocessing_data(df):
    tokenizer=ToktokTokenizer()
    # an inexpensive tokenizer to prioritize speed. 
    # for a more accurate work tokenizer, use word_tokenize from nltk

    nltk.download('stopwords')
    stopword_list = nltk.corpus.stopwords.words('english')



    def rm_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def rm_special_char(text, remove_digits=True):
        pattern = r'[^a-zA-Z0-9\s\[\]]' if remove_digits else r'[^a-zA-Z0-9\s\[\]]'
        text = re.sub(pattern, '', text)
        return text

    def portstem(text):
        # Porter Stemmer, a simple word stemmer. computationally inexpensive.
        # for more advanced stemmers, consider Lancaster or Snowball from nltk.
        # else, can also consider lemmatization to reduce words to a proper base word. (WordNetLemmatizer)
        ps = nltk.porter.PorterStemmer()
        text = ' '.join([ps.stem(word) for word in text.split()])
        return text

    def rm_stopwords(text):
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)    
        return filtered_text


    def preprocess_text(text):
        text = rm_html(text)
        text = rm_special_char(text)
        text = portstem(text)
        text = rm_stopwords(text)
        return text

    df['review'] = df['review'].apply(preprocess_text)

    return df




