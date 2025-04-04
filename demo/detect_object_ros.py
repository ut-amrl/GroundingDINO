import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

import rospy
from std_msgs.msg import Float32MultiArray
import roslib; roslib.load_manifest('amrl_msgs')
from amrl_msgs.msg import *
from amrl_msgs.srv import SemanticObjectDetectionSrv, SemanticObjectDetectionSrvResponse
from sensor_msgs.msg import Image

BOX_THRESHOLD = None
TEXT_THRESHOLD = None
CPU_ONLY = None
MODEL = None
VERBOSE = None

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = PILImage.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    labels, confs = [], []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        labels.append(pred_phrase)
        confs.append(logit.max().item())

    return boxes_filt, pred_phrases, labels, confs

def ros_image_to_pil(ros_image):
    # Convert raw image data to numpy array
    np_arr = np.frombuffer(ros_image.data, dtype=np.uint8)
    # Reshape based on encoding
    if ros_image.encoding == "rgb8":
        image = np_arr.reshape((ros_image.height, ros_image.width, 3))
    elif ros_image.encoding == "bgr8":
        image = np_arr.reshape((ros_image.height, ros_image.width, 3))
        image = image[:, :, ::-1]  # BGR -> RGB
    elif ros_image.encoding == "mono8":
        image = np_arr.reshape((ros_image.height, ros_image.width))
    else:
        raise ValueError(f"Unsupported encoding: {ros_image.encoding}")
    return PILImage.fromarray(image)

def handle_object_detection_request(req):
    rospy.loginfo(f"Received request with query: {req.query_text}")
    
    query_pil_img = ros_image_to_pil(req.query_image)
    query_txt = req.query_text
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    query_image, _ = transform(query_pil_img, None) 
    # run model
    boxes_filt, pred_phrases, labels, confs = get_grounding_output(
        MODEL, query_image, query_txt, BOX_THRESHOLD, TEXT_THRESHOLD, cpu_only=CPU_ONLY, token_spans=eval(f"{TOKEN_SPANS}")
    )
    
    if VERBOSE:
        size = query_pil_img.size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,
        }
        query_pil_img.save(os.path.join("debug", "raw_image.jpg"))
        image_with_box = plot_boxes_to_image(query_pil_img, pred_dict)[0]
        image_with_box.save(os.path.join("debug", "pred.jpg"))
    
    W, H = query_pil_img.size
    bbox_arr_msg = BBox2DArrayMsg(header=req.query_image.header)
    for box, label, conf in zip(boxes_filt, labels, confs):
        box = box * torch.Tensor([W, H, W, H])
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        xyxy = [x0, y0, x1, y1]
        bbox_arr_msg.bboxes.append(BBox2DMsg(label=label, conf=conf, xyxy=xyxy))
    rospy.loginfo(f"Sending results with query: {req.query_text}")
    return SemanticObjectDetectionSrvResponse(bounding_boxes=bbox_arr_msg)

    
def grounding_dino_service():
    rospy.init_node('grounding_dino_bbox_detector_service')
    service = rospy.Service('grounding_dino_bbox_detector', SemanticObjectDetectionSrv, handle_object_detection_request)
    rospy.loginfo("GroundingDINO service is ready.")
    rospy.spin()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")

    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    parser.add_argument("--verbose", "-v", action="store_true", help="debug mode, default=False")
    args = parser.parse_args()

    # cfg
    BOX_THRESHOLD = args.box_threshold
    TEXT_THRESHOLD = args.text_threshold
    TOKEN_SPANS = args.token_spans
    CPU_ONLY = args.cpu_only
    VERBOSE = args.verbose
    
    MODEL = load_model(args.config_file, args.checkpoint_path, cpu_only=CPU_ONLY)
    grounding_dino_service()
