import os

import cv2
import requests
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch

from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .live_portrait_pipeline_img import LivePortraitPipeline


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


class MimicHeadSDK:
    def __init__(self):

        model_save_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../pretrained_weights"
        )
        self.download_models(model_save_folder)
        args = {
            "source_image": "",
            "driving_info": "",
            "output_dir": "animations/",
            "device_id": 0,
            "flag_lip_zero": True,
            "flag_eye_retargeting": False,
            "flag_lip_retargeting": False,
            "flag_stitching": True,
            "flag_relative": True,
            "flag_pasteback": True,
            "flag_do_crop": True,
            "flag_do_rot": True,
            "dsize": 512,
            "scale": 2.3,
            "vx_ratio": 0,
            "vy_ratio": -0.125,
            "server_port": 8890,
            "share": False,
            "server_name": "0.0.0.0",
        }
        inference_cfg = partial_fields(
            InferenceConfig, args
        )  # use attribute of args to initial InferenceConfig
        if not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                inference_cfg.device = "mps"
            else:
                inference_cfg.device = "cpu"
        print(f"==> Use backend {inference_cfg.device}")

        crop_cfg = partial_fields(
            CropConfig, args
        )  # use attribute of args to initial CropConfig
        self.pipeline = LivePortraitPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg,
        )

    def download_models(self, save_folder):
        def download_one_file(
            save_folder,
            sub_path,
            url_prefix="https://modelscope.cn/models/yunfeng/mimic_head/resolve/master/pretrained_weights",
        ):
            whole_path = os.path.join(save_folder, sub_path)
            os.makedirs(os.path.dirname(whole_path), exist_ok=True)
            url = os.path.join(url_prefix, sub_path)
            print(f"==> Downloading {url} to {whole_path}")

            # 使用 requests 模块下载文件
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(whole_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                print("File downloaded successfully")
            else:
                print(
                    f"Failed to download file. HTTP status code: {response.status_code}"
                )

        model_sub_paths = [
            "insightface/models/buffalo_l/2d106det.onnx",
            "insightface/models/buffalo_l/det_10g.onnx",
            "liveportrait/retargeting_models/stitching_retargeting_module.pth",
            "liveportrait/base_models/motion_extractor.pth",
            "liveportrait/base_models/warping_module.pth",
            "liveportrait/base_models/spade_generator.pth",
            "liveportrait/base_models/appearance_feature_extractor.pth",
            "liveportrait/landmark.onnx",
        ]

        for sub_path in model_sub_paths:
            whole_path = os.path.join(save_folder, sub_path)
            if not os.path.isfile(whole_path):
                download_one_file(save_folder, sub_path)

    def process(self, source_img, img):
        if source_img is None:
            return
        self.pipeline.set_source_image(source_img)
        return self.pipeline.process(img)

    def process_video(self, source_img, video_path):
        if source_img is None:
            return
        self.pipeline.set_source_image(source_img)
        if video_path is None:
            return
        cap = cv2.VideoCapture(str(video_path))

        frames = []
        result_img = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB for display in Gradio
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Invert colors
            result_img = self.pipeline.process(frame)

            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            # Append the inverted frame to the list of frames
            frames.append(result_img)

        cap.release()

        # Combine all frames into one video
        out = cv2.VideoWriter(
            "output.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (result_img.shape[1], result_img.shape[0]),
        )
        for f in frames:
            out.write(f)
        out.release()

        # Return the path to the output video file
        return "output.mp4"
