from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

def multinb(X_train, X_test, y_train, y_test):
    from sklearn.naive_bayes import MultinomialNB
    
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return clf