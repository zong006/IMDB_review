A simple NLP (natural language processing) machine learning project on a classification task with a balanced target class. 


#### DataSet 
The dataset used here for sentiment analysis of movie reviews is from the following Kaggle URL :

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

- review: text string of movie reviews in english, with the presence of some html style text and special characters.
- sentiment: positive or negative.


#### Text Preprocessing:

The steps for text preprocessing outlined here are to prioritize speed.

1. HTML syntax: BeautifulSoup is used for parsing and removing HTML syntax in the strings
2. Special characters: these are removed with regex
3. Word stemming: Porter Stemmer, a simple word stemmer which is computationally inexpensive is used. For more advanced stemmers, consider Lancaster or Snowball from nltk. Else we can also consider lemmatization to reduce words to a proper base word. (WordNetLemmatizer)
4. Tokenize:  A simple tokenizer should be sufficient and ToktokTokenizer, an inexpensive tokenizer is used. For a potentially more accurate work tokenizer, use word_tokenize from nltk.
5. Stopwords: A stopword list is downloaded from nltk, and words in the list are filtered out from the tokenized texts.

Lastly, we have to convert the strings into matrices as inputs to the machine learning algorithm via vectorization. Two common vectorizers are Count vectorizer, and TFIDF vectorizer. The latter is more expensive but can better capture the importance of certain words. Stopwords have been removed to potentially speed up any vectorization process, so we shall use TFIDF.

The dataset is first split into train/test sets, and the vectorizer is fitted on the train set. After performing a fit transform on the train set, the fitted vectorizer is then applied to the test set. 


#### Description of logical steps/flow of the pipeline

1. The raw data is a .csv file, train.csv.
2. Data extraction from .csv files are performed by the script data_extraction.py, giving a pandas dataframe as an output.
3. The dataframe is fed into the script data_preprocessing.py, where text strings are processed using steps 1-5 outlined above in "Text Preprocessing". The script outputs the processed dataframe.
3. The data vectorized using TFIDF and fed into algo.py containing machine learning algorithms which print classification reports for the classification task and outputs the classifier for in this script. 


#### Choice of models and evaluations

This is a classification problem with balanced binary target class. In particular, considering that the inputs are transformed into sparse matrices after preprocessing and vectorization, a simple algorithm suitable for such a task is a multinomial Naive-Bayes. It is a commonly used algorithm for text classification tasks, including sentiment analysis and is well-suited for handling such sparse data efficiently. Its also computationally efficient and scales well with large datasets. 


- The overall accuracy is 0.74, with a precision of 0.75 and 0.73 for the positive and negative classes, respectively.

Given that very simple preprocessing methods were used (since computational speed was being prioritized), the model gives a reasonable performance. We can however experiment with more expensive preprocessing methods as mentioned in "Text Preprocessing", as well as use more sophisticated algorithms like random forest classifiers at the expense of speed.
    


