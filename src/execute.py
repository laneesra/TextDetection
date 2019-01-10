import feature_extraction
import clf
import text_localizing


feature_extraction.run()
clf.predict_catboost()
text_localizing.text_localizing()
