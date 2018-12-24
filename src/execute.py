import feature_extraction
import TextDetection.clf
import text_localizing


id = input()
feature_extraction.run(id)
TextDetection.clf.predict_catboost(id)
text_localizing.text_localizing(id)
