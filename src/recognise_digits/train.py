from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model():
    X, y = load_digits(return_X_y=True, as_frame=True)
    model = SVC()

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    return accuracy_score(ytest, ypred)