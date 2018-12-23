import feature_extraction, clf, text_localizing

id = '0469'
feature_extraction.run(id)
clf.predict_catboost(id)
text_localizing.text_localizing(id)
