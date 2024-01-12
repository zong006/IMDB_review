import pandas as pd
from data_extraction import extract_data
from data_preprocessing import pre_processing_data
from algo import multinb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer    
    
    
def main():

    df = extract_data()
    df = pre_processing_data(df)

    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

    tv = TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
    X_train = tv.fit_transform(X_train)
    X_test = tv.transform(X_test)

    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)

    clf_nb = multinb(X_train, X_test, y_train, y_test)

    return 

if __name__ == "__main__":
    main()