"""
Trains a small classifier on the synthetic vision_data.npz (drawer
open/closed from a downsampled camera image), then converts it to OpenVINO
IR for edge inference.

This is a genuinely small, genuinely trained model -- a logistic regression
over downsampled pixel values, not a large vision backbone. Scope chosen
deliberately: real camera-based reasoning, honestly sized for the time
available, converted through a real OpenVINO pipeline rather than a larger
model that would only be claimed, not actually run. See README.md.

Usage: python train_vision_model.py
Produces: vision_model.onnx, vision_model_ir/ (OpenVINO IR: .xml + .bin)
"""
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "vision_data.npz")
ONNX_PATH = os.path.join(HERE, "vision_model.onnx")
IR_DIR = os.path.join(HERE, "vision_model_ir")


def main():
    data = np.load(DATA_PATH)
    images, labels = data["images"], data["labels"]

    # Flatten + normalize to [0,1] -- the whole feature vector is
    # (48*64*3 =) 9216 downsampled pixel values, small enough for a fast,
    # genuinely-trainable-in-seconds logistic regression.
    X = images.reshape(len(images), -1).astype(np.float32) / 255.0
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Train accuracy: {train_acc:.3f}  |  Test accuracy: {test_acc:.3f}")

    # --- Convert to ONNX: hand-built standard-ops graph, not skl2onnx's
    # default ai.onnx.ml.LinearClassifier -- OpenVINO's ONNX frontend
    # doesn't implement the ai.onnx.ml operator set at all (confirmed
    # in-session: conversion fails with "No conversion rule found for
    # operations: ai.onnx.ml.LinearClassifier"). Logistic regression is
    # just sigmoid(X @ W.T + b), so building that directly with standard
    # ops (Gemm + Sigmoid) sidesteps the whole problem. ---
    import onnx.helper as oh
    from onnx import TensorProto

    W = clf.coef_.astype(np.float32)          # shape (1, n_features)
    b = clf.intercept_.astype(np.float32)      # shape (1,)
    n_features = W.shape[1]

    W_init = oh.make_tensor("W", TensorProto.FLOAT, W.shape, W.flatten().tolist())
    b_init = oh.make_tensor("b", TensorProto.FLOAT, b.shape, b.flatten().tolist())

    input_tensor = oh.make_tensor_value_info("input", TensorProto.FLOAT, [None, n_features])
    output_tensor = oh.make_tensor_value_info("probability", TensorProto.FLOAT, [None, 1])

    gemm_node = oh.make_node(
        "Gemm", inputs=["input", "W", "b"], outputs=["logits"],
        alpha=1.0, beta=1.0, transB=1,
    )
    sigmoid_node = oh.make_node("Sigmoid", inputs=["logits"], outputs=["probability"])

    graph = oh.make_graph(
        [gemm_node, sigmoid_node], "drawer_state_classifier",
        [input_tensor], [output_tensor], initializer=[W_init, b_init],
    )
    onnx_model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 13)])
    onnx.checker.check_model(onnx_model)
    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Saved ONNX model: {ONNX_PATH}")

    # Sanity-check the hand-built graph matches sklearn's own predictions
    # before trusting the OpenVINO conversion of it.
    import onnxruntime as _ort_check
    sess = _ort_check.InferenceSession(ONNX_PATH)
    onnx_probs = sess.run(None, {"input": X_test[:20]})[0].flatten()
    sklearn_probs = clf.predict_proba(X_test[:20])[:, 1]
    max_diff = np.abs(onnx_probs - sklearn_probs).max()
    print(f"Hand-built ONNX graph vs sklearn max prob diff (first 20 test samples): {max_diff:.6f}")
    assert max_diff < 1e-4, "ONNX graph does not match sklearn's own predictions"

    # --- Convert ONNX -> OpenVINO IR ---
    import openvino as ov
    ov_model = ov.convert_model(ONNX_PATH)
    os.makedirs(IR_DIR, exist_ok=True)
    ov.save_model(ov_model, os.path.join(IR_DIR, "vision_model.xml"))
    print(f"Saved OpenVINO IR: {IR_DIR}/vision_model.xml (+ .bin)")

    return test_acc


if __name__ == "__main__":
    main()