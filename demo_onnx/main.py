import sys
import argparse
import math
import numpy as np
import cv2 as cv
import onnxruntime as ort


MEAN = [103.53, 116.28, 123.675]
STD = [57.375, 57.12, 58.395]

CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.3

STRIDES = [8, 16, 32, 64]


def get_resize_matrix(raw_shape, dst_shape, keep_ratio):
    r_h, r_w = raw_shape
    d_h, d_w = dst_shape
    Rs = np.eye(3)
    if keep_ratio:
        C = np.eye(3)
        C[0, 2] = -r_w / 2
        C[1, 2] = -r_h / 2

        if r_w / r_h < d_w / d_h:
            ratio = d_h / r_h
        else:
            ratio = d_w / r_w
        Rs[0, 0] *= ratio
        Rs[1, 1] *= ratio

        T = np.eye(3)
        T[0, 2] = 0.5 * d_w
        T[1, 2] = 0.5 * d_h
        return T @ Rs @ C
    else:
        Rs[0, 0] *= d_w / r_w
        Rs[1, 1] *= d_h / r_h
        return Rs


def preprocess(img, input_size):
    img_size = img.shape[:2]

    # TODO: Prefer OpenCV resize; warpPerspective is used only to match NanoDet-Plus preprocessing exactly.
    M = get_resize_matrix(img_size, input_size, False)
    img = cv.warpPerspective(img, M, dsize=input_size)

    img = img.astype(np.float32) / 255.0
    mean = np.array(MEAN, dtype="float32").reshape(1, 1, -1) / 255.0
    std = np.array(STD, dtype="float32").reshape(1, 1, -1) / 255.0
    img = (img - mean) / std
    img = np.moveaxis(img, -1, 0)
    return img[np.newaxis, ...]


def get_priors(fmap_size, stride):
    h, w = fmap_size
    grid_x = np.arange(w) * stride
    grid_y = np.arange(h) * stride
    y, x = np.meshgrid(grid_y, grid_x, indexing="ij")
    y = y.flatten()
    x = x.flatten()
    s = np.full(x.shape, stride)
    grid = np.stack((x, y, s, s), axis=-1)
    return grid


def softmax(logits):
    exp_logits = np.exp(logits)
    sum_exp_logits = np.sum(exp_logits, axis=-1, keepdims=True)
    probs = exp_logits / sum_exp_logits
    return probs


def integrate(reg_preds, reg_max=7):
    project = np.linspace(0, reg_max, reg_max + 1)
    reg_preds = reg_preds.reshape(*reg_preds.shape[:-1], 4, -1)
    probs = softmax(reg_preds)
    dist = probs @ project.T
    return dist


def distance_to_box(center, dist, max_size):
    x1 = center[..., 0] - dist[..., 0]
    y1 = center[..., 1] - dist[..., 1]
    x2 = center[..., 0] + dist[..., 2]
    y2 = center[..., 1] + dist[..., 3]
    max_h, max_w = max_size
    x1 = np.clip(x1, 0, max_w)
    y1 = np.clip(y1, 0, max_h)
    x2 = np.clip(x2, 0, max_w)
    y2 = np.clip(y2, 0, max_h)
    return np.stack((x1, y1, x2, y2), axis=-1)


def post_process(output, input_size, n_classes):
    scores = output[..., :n_classes]
    reg_preds = output[..., n_classes:]

    fmap_sizes = [
        (math.ceil(input_size[1] / s), math.ceil(input_size[0] / s)) for s in STRIDES
    ]
    priors = [get_priors(fsize, s) for fsize, s in zip(fmap_sizes, STRIDES)]
    priors = np.vstack(priors)

    dist = integrate(reg_preds) * priors[..., 2, np.newaxis]
    boxes = distance_to_box(priors[..., :2], dist, input_size)
    confs = scores.max(axis=-1)
    labels = scores.argmax(axis=-1)
    idx = confs >= CONF_THRESHOLD
    boxes = boxes[idx]
    confs = confs[idx]
    labels = labels[idx]
    return nms(boxes, confs, labels)


def nms(boxes, confs, labels, iou_threshold=IOU_THRESHOLD):
    idx = np.argsort(confs)[::-1]
    boxes = boxes[idx]
    confs = confs[idx]
    labels = labels[idx]

    out_boxes = []
    out_confs = []
    out_labels = []

    while idx.size > 0:
        i = idx[0]
        out_boxes.append(boxes[i])
        out_confs.append(confs[i])
        out_labels.append(labels[i])

        if idx.size == 1:
            break
        iou_scores = iou(
            boxes[i].reshape(-1, 4), boxes[idx[1:]].reshape(-1, 4)
        ).reshape(-1)
        ind = np.nonzero(iou_scores <= iou_threshold)[0]
        idx = idx[ind + 1]
    return out_boxes, out_confs, out_labels


def iou(boxes1, boxes2):
    area = lambda boxes: (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    a1 = area(boxes1)
    a2 = area(boxes2)
    inter_upperlefts = np.maximum(boxes1[:, np.newaxis, :2], boxes2[:, :2])
    inter_lowerrights = np.minimum(boxes1[:, np.newaxis, 2:], boxes2[:, 2:])
    inter_wh = (inter_lowerrights - inter_upperlefts).clip(0, None)
    inter_a = inter_wh[..., 0] * inter_wh[..., 1]
    union_a = a1[..., np.newaxis] + a2 - inter_a
    return inter_a / union_a


def run_infer(onnx_model, input_size, n_classes, in_path, out_path):
    img = cv.imread(in_path)
    if img is None:
        raise "Invalid image"

    img_size = img.shape[:2]
    img_input = preprocess(img, input_size)

    session = ort.InferenceSession(onnx_model)
    outputs = session.run(None, {"data": img_input})
    boxes, confs, labels = post_process(outputs[0], input_size, n_classes)

    x_scale = img_size[1] / input_size[1]
    y_scale = img_size[0] / input_size[0]

    for (x1, y1, x2, y2), conf, label in zip(boxes, confs, labels):
        x1 = int(x1 * x_scale)
        y1 = int(y1 * y_scale)
        x2 = int(x2 * x_scale)
        y2 = int(y2 * y_scale)
        print(f"box: [{x1},{y1},{x2},{y2}] | conf: {conf} | label: {label}")
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.imwrite(out_path, img)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="ONNX inference demo",
    )
    parser.add_argument("-m", "--model", type=str, help="path to ONNX model")
    parser.add_argument(
        "-s", "--input_size", type=str, help="h,w - input size to the model"
    )
    parser.add_argument("-n", "--n_classes", type=int, help="number of classes")
    parser.add_argument("-i", "--input", type=str, help="path to input image")
    parser.add_argument("-o", "--output", type=str, help="path to write output")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_infer(
        args.model, eval(args.input_size), args.n_classes, args.input, args.output
    )
