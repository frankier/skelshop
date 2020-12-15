from sklearn.model_selection import GridSearchCV


class GridSearchClus(GridSearchCV):
    def __init__(self, **kwargs):
        super().__init__(
            # Disable cross validation
            cv=[(slice(None), slice(None))],
            **kwargs
        )
