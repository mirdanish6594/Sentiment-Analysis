# xgboost_wrapper.py
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from types import SimpleNamespace

class XGBWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted_ = True
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.model.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.model.predict_proba(X)

    def score(self, X, y):
        check_is_fitted(self)
        return self.model.score(X, y)

    def _get_tags(self):
        return SimpleNamespace(
            requires_y=True,
            non_deterministic=False,
            requires_positive_X=False,
            requires_positive_y=False,
            X_types=["2darray"],
            poor_score=False,
            no_validation=False,
            multioutput=False,
            allow_nan=True,
            stateless=False,
            multilabel=False,
            _skip_test=False,
            _xfail_checks=False,
            multioutput_only=False,
            binary_only=False,
            requires_fit=True,
        )

    def __sklearn_tags__(self):
        return self._get_tags()