import joblib
def predict(data):

    clf = joblib.load('xgb_classification.sav')

    class_result = clf.predict(data)
    class_label = ''

    if class_result == 0:
        class_label = 'Low'
    elif class_result == 1:
        class_label = 'Medium'
    else:
        class_label = 'High'

    return class_label
