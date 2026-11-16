model = LogisticRegression()
atalas = 0.85

from sklearn.model_selection import FixedThresholdClassifier

model_t = FixedThresholdClassifier(model, threshold=atalas, response_method="predict_proba")