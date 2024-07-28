import cv2
import numpy as np

from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.io import resize_to_limit
from .live_portrait_wrapper import LivePortraitWrapper


class LivePortraitPipeline(object):
    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.inference_cfg = inference_cfg
        self.crop_cfg = crop_cfg
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(
            cfg=inference_cfg
        )
        self.cropper = Cropper(crop_cfg=crop_cfg)
        self.R_d_0, self.x_d_0_info = None, None

        self.source_image = None

    def set_source_image(self, img_rgb_raw):
        if img_rgb_raw is None:
            return
        if (
            self.source_image is not None
            and img_rgb_raw.shape == self.source_image.shape
        ):
            if np.all(img_rgb_raw == self.source_image):
                return

        img_rgb = resize_to_limit(
            img_rgb_raw,
            self.inference_cfg.ref_max_shape,
            self.inference_cfg.ref_shape_n,
        )
        crop_info = self.cropper.crop_single_image(img_rgb)
        source_lmk = crop_info["lmk_crop"]
        img_crop, img_crop_256x256 = (
            crop_info["img_crop"],
            crop_info["img_crop_256x256"],
        )
        I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
        self.x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)
        self.x_c_s = self.x_s_info["kp"]
        self.R_s = get_rotation_matrix(
            self.x_s_info["pitch"], self.x_s_info["yaw"], self.x_s_info["roll"]
        )
        self.f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
        self.x_s = self.live_portrait_wrapper.transform_keypoint(self.x_s_info)
        c_d_lip_before_animation = [0.0]
        combined_lip_ratio_tensor_before_animation = (
            self.live_portrait_wrapper.calc_combined_lip_ratio(
                c_d_lip_before_animation, source_lmk
            )
        )
        self.lip_delta_before_animation = self.live_portrait_wrapper.retarget_lip(
            self.x_s, combined_lip_ratio_tensor_before_animation
        )

        self.source_image = img_rgb_raw

    def process(self, img):
        if img is None:
            return
        inference_cfg = self.live_portrait_wrapper.cfg  # for convenience

        driving_rgb_lst = [img]
        driving_rgb_lst_256 = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]
        I_d_lst = self.live_portrait_wrapper.prepare_driving_videos(driving_rgb_lst_256)

        I_d_i = I_d_lst[0]
        x_d_i_info = self.live_portrait_wrapper.get_kp_info(I_d_i)
        R_d_i = get_rotation_matrix(
            x_d_i_info["pitch"], x_d_i_info["yaw"], x_d_i_info["roll"]
        )

        if self.R_d_0 is None:
            self.R_d_0 = R_d_i
            self.x_d_0_info = x_d_i_info

        R_new = (R_d_i @ self.R_d_0.permute(0, 2, 1)) @ self.R_s
        delta_new = self.x_s_info["exp"] + (x_d_i_info["exp"] - self.x_d_0_info["exp"])
        scale_new = self.x_s_info["scale"] * (
            x_d_i_info["scale"] / self.x_d_0_info["scale"]
        )
        t_new = self.x_s_info["t"] + (x_d_i_info["t"] - self.x_d_0_info["t"])

        x_d_i_new = scale_new * (self.x_c_s @ R_new + delta_new) + t_new
        x_d_i_new = self.live_portrait_wrapper.stitching(
            self.x_s, x_d_i_new
        ) + self.lip_delta_before_animation.reshape(-1, self.x_s.shape[1], 3)

        out = self.live_portrait_wrapper.warp_decode(self.f_s, self.x_s, x_d_i_new)
        I_p_i = self.live_portrait_wrapper.parse_output(out["out"])[0]
        return I_p_i

    def process_img(self, img):
        if img is None:
            return
        inference_cfg = self.live_portrait_wrapper.cfg  # for convenience

        driving_rgb_lst = [img]
        driving_rgb_lst_256 = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]
        I_d_lst = self.live_portrait_wrapper.prepare_driving_videos(driving_rgb_lst_256)

        I_d_i = I_d_lst[0]
        x_d_i_info = self.live_portrait_wrapper.get_kp_info(I_d_i)
        R_d_i = get_rotation_matrix(
            x_d_i_info["pitch"], x_d_i_info["yaw"], x_d_i_info["roll"]
        )

        R_new = R_d_i @ self.R_s
        delta_new = self.x_s_info["exp"] + x_d_i_info["exp"]
        scale_new = self.x_s_info["scale"] * x_d_i_info["scale"]
        t_new = self.x_s_info["t"] + x_d_i_info["t"]

        x_d_i_new = scale_new * (self.x_c_s @ R_new + delta_new) + t_new
        x_d_i_new = self.live_portrait_wrapper.stitching(
            self.x_s, x_d_i_new
        ) + self.lip_delta_before_animation.reshape(-1, self.x_s.shape[1], 3)

        out = self.live_portrait_wrapper.warp_decode(self.f_s, self.x_s, x_d_i_new)
        I_p_i = self.live_portrait_wrapper.parse_output(out["out"])[0]
        return I_p_i
