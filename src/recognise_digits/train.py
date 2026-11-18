from utils import get_model
from joblib import dump
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model():
    X, y = fetch_openml(
        'mnist_784',
        version=1,
        return_X_y=True,
        as_frame=False,
        parser='liac-arff'
    )
    model = SVC()

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    dump(model, get_model())
    # print(accuracy_score(ytest, ypred))