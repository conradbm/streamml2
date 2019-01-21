class AbstractFeatureSelectionModel:
    _X=None
    _y=None
    _verbose=None
    _params=None
    _code=None
    def __init__(self,code, X, y, params, verbose):
        self._code=code
        self._X=X
        self._y=y
        self._params=params
        self._verbose=verbose
        
    def execute(self):
        assert isinstance(self._code,str), "code must be of type str."
        assert not self._X is None, "please provide a valid data set, perhaps you didn't construct the model before calling this function."
        assert not self._y is None, "please provide a valid response set, perhaps you didn't construct the model before calling this function."
        assert isinstance(self._params, dict), "please ensure that params are of type dict."
        assert isinstance(self._verbose,bool), "verbose must be of type bool."
        
        if self._verbose:
            print("Executing: " + self._code)