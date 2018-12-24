import feature_extraction
import clf
import text_localizing


id = raw_input()
feature_extraction.run(id)
clf.predict_catboost(id)
text_localizing.text_localizing(id)
