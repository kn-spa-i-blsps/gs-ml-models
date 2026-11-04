# Splat Classifier API

## API

Each model file or submodule should contain a named model class that can be
initialized with different parameters where applicable. The following methods
should be exposed:

- `fit(X, y)` trains the model from scratch on the provided images and labels.
  `X` has shape `(B, 3, 480, 640)`, where `B` is the number of images, and `y`
  has shape `(B, 4)`.
- `train(X, y)` continues training an existing model on additional data without
  resetting it. The inputs follow the same format as for `fit`.
- `predict(X)` returns an array of shape `(B, 4)` with four values between 0 and 1 for each image.
- `save(filename)` saves a representation of the model that can be restored with `load`. A model may use a directory instead of a single file if needed.
- `load(filename)` returns an instance of the model class restored from a
  representation produced by `save`.

### Optional methods

Add optional methods here to keep their interfaces consistent across models.

- `predict_extra(X)` returns any additional model output beyond the four predictions. Its output may be a NumPy array, Torch tensor, or a regular Python object.
