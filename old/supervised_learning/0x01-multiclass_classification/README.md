# 0x01 - Multiclass Classification

## Learning Objectives

- What is multiclass classification?
- What is a one-hot vector?
- How to encode/decode one-hot vectors
- What is the softmax function and when do you use it?
- What is cross-entropy loss?
- What is pickling in Python?

---

### [0. One-Hot Encode](./0-one_hot_encode.py)

Write a function `one_hot_encode(Y, classes)` that converts a numeric label vector into a one-hot matrix:
- Y is a `numpy.ndarray` with shape `(m,)` containing numeric class labels
  - `m` is the number of examples
- `classes` is the maximum number of classes found in `Y`
- Returns a one-hot encoding of `Y` with shape `(classes, m)`, or `None` on failure

---

## Author

[**Patrick DeYoreo**](github.com/patrickdeyoreo)
