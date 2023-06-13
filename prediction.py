import joblib
def predict(data):

    rlf = joblib.load('./rf_regression.sav')
    clf = joblib.load('./xgb_classification.sav')

    reg_result = rlf.predict(data)

    class_result = clf.predict(data)
    class_label = ''

    if class_result == 0:
        class_label = 'Low'
    elif class_result == 1:
        class_label = 'Medium'
    else:
        class_label = 'High'

    return reg_result, class_label
