# Anomaly Detection

## API

The model files (or submodule) should contain a named model class
that can be initialized with different parameters (if applicable). \
The following methods should be exposed:

- `fit(X)` trains the model from scratch on the data provided in X. Note that there are only features because we are doing unsupervised learning. The input X is of shape (B, 3, 480, 640) where B represents the number of batches.

- `train(X)` trains the existing model on the additional data without clearning it (basically allows to make a model better from an existing model)

- `predict(X)` for X of shape (B, 3, 480, 640) returns an array of shape (B,) which contains 0/1 labels for each input, 1 if an anomaly is detected, 0 if not.

- `anomaly_scores(X)` same as X but rather than returning a discrete classification returns an array of shape (B,) which contains the model's scores. In some cases, this is not applicable, but should be provided if possible.

- `save(filename)` saves some representation of the model to the given file. The only real requirement is that it's compatible with load. This can be changed a little if the model saves a folder rather than a single file for some reason.

- `load(filename)` returns an instance of the model class loaded from the provided file. Pretty much only needs to be compatible with save().

### Optional methods

If you are looking to add something to the API or implement a new method, put it here to maintain consistency with others' methods

List of optional methods:

- `predict_extra(X)` should return some sort of representation for your predictions in case the model produces some more sophisticated outputs than just scores or labels (for example if the model identifies anomaly locations). THe output here doesn't have to be a numpy array / torch tensor and might just be a "regular" list of python objects.
