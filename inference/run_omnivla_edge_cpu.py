"""
run_omnivla_edge_cpu.py
────────────────────────
Versión adaptada de run_omnivla_edge.py para correr en CPU.
Usa OpenVINO si el modelo fue convertido, si no usa PyTorch CPU.

Uso:
    python inference/run_omnivla_edge_cpu.py

Para convertir el modelo primero (recomendado, más rápido):
    python inference/convert_to_openvino.py
"""

import sys, os
sys.path.insert(0, '.')

import time, math, json
from typing import Optional, Tuple, Dict
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import utm
import clip

from utils_policy import transform_images_map, load_model, transform_images_PIL, transform_images_PIL_mask

# ── Rutas ────────────────────────────────────────────────────────
MODEL_WEIGHTS_PATH = "./omnivla-edge"
OV_MODEL_PATH      = "./omnivla-edge-ov/omnivla_edge.xml"
ONNX_MODEL_PATH    = "./omnivla-edge-ov/omnivla_edge.onnx"

# ─────────────────────────────────────────────────────────────────
# Backend de inferencia — elige automáticamente el mejor disponible
# ─────────────────────────────────────────────────────────────────
class InferenceBackend:
    """
    Jerarquía de backends (de más rápido a más lento en CPU):
      1. OpenVINO IR   (.xml) — 2-3x más rápido que PyTorch en CPU Intel
      2. ONNX Runtime  (.onnx) — ~1.5x más rápido que PyTorch CPU
      3. PyTorch CPU   (.pth)  — fallback siempre disponible
    """
    def __init__(self, model_params, device):
        self.backend  = None
        self.ov_model = None
        self.ort_sess = None
        self.pt_model = None
        self.device   = device

        if os.path.exists(OV_MODEL_PATH):
            self._load_openvino()
        elif os.path.exists(ONNX_MODEL_PATH):
            self._load_onnxruntime()
        else:
            self._load_pytorch(model_params)

    def _load_openvino(self):
        try:
            import openvino as ov
            core = ov.Core()
            # Optimizaciones para CPU Intel
            core.set_property("CPU", {
                "PERFORMANCE_HINT": "LATENCY",
                "INFERENCE_NUM_THREADS": str(os.cpu_count() or 4),
                "ENABLE_CPU_PINNING": "YES",
            })
            self.ov_model = core.compile_model(OV_MODEL_PATH, "CPU")
            self.backend  = "openvino"
            print(f"[Backend] OpenVINO IR — {OV_MODEL_PATH}")
            print(f"          CPU threads: {os.cpu_count()}")
        except Exception as e:
            print(f"[Backend] OpenVINO falló ({e}), intentando ONNX...")
            if os.path.exists(ONNX_MODEL_PATH):
                self._load_onnxruntime()
            else:
                self._load_pytorch({})

    def _load_onnxruntime(self):
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = os.cpu_count() or 4
            opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            self.ort_sess = ort.InferenceSession(
                ONNX_MODEL_PATH,
                sess_options=opts,
                providers=["CPUExecutionProvider"]
            )
            self.backend = "onnxruntime"
            print(f"[Backend] ONNX Runtime — {ONNX_MODEL_PATH}")
        except Exception as e:
            print(f"[Backend] ONNX Runtime falló ({e}), usando PyTorch CPU...")
            self._load_pytorch({})

    def _load_pytorch(self, model_params):
        ckpth_path = os.path.join(MODEL_WEIGHTS_PATH, "omnivla-edge.pth")
        if not os.path.exists(ckpth_path):
            raise FileNotFoundError(
                f"No se encontró el modelo en {ckpth_path}\n"
                "Asegúrate de tener la carpeta omnivla-edge/ con el archivo .pth"
            )
        self.pt_model, _, _ = load_model(ckpth_path, model_params, self.device)
        self.pt_model = self.pt_model.to(self.device).eval()
        self.backend  = "pytorch_cpu"
        print(f"[Backend] PyTorch CPU — {ckpth_path}")
        print("          Tip: ejecuta convert_to_openvino.py para acelerar 2-3x")

    def run(self, obs_images, goal_pose, map_images, goal_image,
            modality_id, feat_text, cur_large_img):
        """Ejecuta el forward pass en el backend disponible."""

        if self.backend == "openvino":
            result = self.ov_model({
                "obs_images":    obs_images.numpy(),
                "goal_pose":     goal_pose.numpy(),
                "map_images":    map_images.numpy(),
                "goal_image":    goal_image.numpy(),
                "modality_id":   modality_id.numpy(),
                "feat_text":     feat_text.numpy(),
                "cur_large_img": cur_large_img.numpy(),
            })
            actions = torch.from_numpy(list(result.values())[0])
            mask    = torch.from_numpy(list(result.values())[2])
            return actions, modality_id, mask

        elif self.backend == "onnxruntime":
            result = self.ort_sess.run(None, {
                "obs_images":    obs_images.numpy(),
                "goal_pose":     goal_pose.numpy(),
                "map_images":    map_images.numpy(),
                "goal_image":    goal_image.numpy(),
                "modality_id":   modality_id.numpy(),
                "feat_text":     feat_text.numpy(),
                "cur_large_img": cur_large_img.numpy(),
            })
            actions = torch.from_numpy(result[0])
            mask    = torch.from_numpy(result[2])
            return actions, modality_id, mask

        else:  # pytorch_cpu
            with torch.no_grad():
                predicted_actions, distances, mask_number = self.pt_model(
                    obs_images, goal_pose, map_images,
                    goal_image, modality_id, feat_text, cur_large_img
                )
            return predicted_actions, modality_id, mask_number


# ─────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────
def remove_ddp_in_checkpoint(state_dict):
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

def clip_angle(angle):
    while angle > math.pi:  angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle


# ─────────────────────────────────────────────────────────────────
# Clase principal de inferencia
# ─────────────────────────────────────────────────────────────────
class Inference:
    def __init__(self, save_dir, lan_inst_prompt, goal_utm, goal_compass, goal_image_PIL):
        self.tick_rate       = 3
        self.lan_inst_prompt = lan_inst_prompt
        self.goal_utm        = goal_utm
        self.goal_compass    = goal_compass
        self.goal_image_PIL  = goal_image_PIL
        self.count_id        = 0
        self.linear          = 0.0
        self.angular         = 0.0
        self.datastore_path_image = save_dir

    @staticmethod
    def calculate_relative_position(x_a, y_a, x_b, y_b):
        return x_b - x_a, y_b - y_a

    @staticmethod
    def rotate_to_local_frame(delta_x, delta_y, heading_a_rad):
        rel_x =  delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
        rel_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
        return rel_x, rel_y

    def run(self):
        loop_time  = 1 / self.tick_rate
        start_time = time.time()
        while True:
            if time.time() - start_time > loop_time:
                self.tick()
                start_time = time.time()
                break

    def tick(self):
        self.linear, self.angular = self.run_omnivla()

    def run_omnivla(self):
        thres_dist              = 30.0
        metric_waypoint_spacing = 0.1

        # GPS actual (datos de ejemplo)
        current_lat     = 37.87371258374039
        current_lon     = -122.26729417226024
        current_compass = 270.0
        cur_utm         = utm.from_latlon(current_lat, current_lon)
        cur_compass     = -float(current_compass) / 180.0 * math.pi

        delta_x, delta_y   = self.calculate_relative_position(
            cur_utm[0], cur_utm[1], self.goal_utm[0], self.goal_utm[1]
        )
        relative_x, relative_y = self.rotate_to_local_frame(delta_x, delta_y, cur_compass)
        radius = np.sqrt(relative_x**2 + relative_y**2)
        if radius > thres_dist:
            relative_x *= thres_dist / radius
            relative_y *= thres_dist / radius

        goal_pose_torch = torch.from_numpy(np.array([
            relative_y / metric_waypoint_spacing,
            -relative_x / metric_waypoint_spacing,
            np.cos(self.goal_compass - cur_compass),
            np.sin(self.goal_compass - cur_compass)
        ])).unsqueeze(0).float()

        # Sobreescritura para prueba (comentar para usar GPS real)
        yaw_ang         = -90.0
        goal_pose_torch = torch.from_numpy(np.array([
            1.0 / metric_waypoint_spacing,
            -10.0 / metric_waypoint_spacing,
            np.cos(yaw_ang / 180.0 * 3.1415),
            np.sin(yaw_ang / 180.0 * 3.1415)
        ])).unsqueeze(0).float()

        # Imagen actual
        current_image_PIL     = Image.open("./inference/current_img.jpg").convert("RGB")
        current_image_PIL_96  = current_image_PIL.resize(imgsize)
        current_image_PIL_224 = current_image_PIL.resize(imgsize_clip)

        mask_360_pil_96  = np.ones((96,  96,  3), dtype=np.float32)
        mask_360_pil_224 = np.ones((224, 224, 3), dtype=np.float32)

        context_queue = [current_image_PIL_96] * 6
        obs_images    = transform_images_PIL_mask(context_queue, mask_360_pil_96)
        obs_images    = torch.cat(torch.split(obs_images, 3, dim=1), dim=1)
        obs_image_cur = torch.split(obs_images, 3, dim=1)[-1]

        cur_large_img = transform_images_PIL_mask(current_image_PIL_224, mask_360_pil_224)

        satellite_cur  = Image.new("RGB", (352, 352), color=(0, 0, 0))
        satellite_goal = Image.new("RGB", (352, 352), color=(0, 0, 0))
        current_map    = transform_images_map(satellite_cur)
        goal_map       = transform_images_map(satellite_goal)
        map_images     = torch.cat((current_map, goal_map, obs_image_cur), axis=1)

        lan_inst        = self.lan_inst_prompt if lan_prompt else "xxxx"
        obj_inst_lan    = clip.tokenize(lan_inst, truncate=True)
        feat_text_lan   = text_encoder.encode_text(obj_inst_lan)

        goal_image = transform_images_PIL_mask(goal_image_PIL, mask_360_pil_96)

        # Calcular modality_id
        if   not pose_goal and not satellite and not image_goal and lan_prompt:     mid = 7
        elif not pose_goal and not satellite and image_goal     and not lan_prompt: mid = 6
        elif     pose_goal and not satellite and not image_goal and not lan_prompt: mid = 4
        elif not pose_goal and not satellite and image_goal     and lan_prompt:     mid = 9
        elif     pose_goal and not satellite and image_goal     and not lan_prompt: mid = 5
        elif     pose_goal and not satellite and not image_goal and lan_prompt:     mid = 8
        else: mid = 7

        modality_id = torch.tensor([mid])

        # ── Forward pass con el backend elegido ──────────────────
        t0 = time.time()
        predicted_actions, modality_id, mask_number = backend.run(
            obs_images, goal_pose_torch, map_images,
            goal_image, modality_id, feat_text_lan, cur_large_img
        )
        elapsed = time.time() - t0
        print(f"[Inferencia] {elapsed:.2f}s  |  backend: {backend.backend}")
        print("Generated action chunk", predicted_actions)

        waypoints       = predicted_actions.float().numpy()
        waypoint_select = 4
        chosen_waypoint = waypoints[0][waypoint_select].copy()
        chosen_waypoint[:2] *= metric_waypoint_spacing
        dx, dy, hx, hy  = chosen_waypoint

        EPS = 1e-8
        DT  = 1 / 3
        if np.abs(dx) < EPS and np.abs(dy) < EPS:
            linear_vel  = 0
            angular_vel = 1.0 * clip_angle(np.arctan2(hy, hx)) / DT
        elif np.abs(dx) < EPS:
            linear_vel  = 0
            angular_vel = 1.0 * np.sign(dy) * np.pi / (2 * DT)
        else:
            linear_vel  = dx / DT
            angular_vel = np.arctan(dy / dx) / DT

        linear_vel  = float(np.clip(linear_vel,  0,    0.5))
        angular_vel = float(np.clip(angular_vel, -1.0, 1.0))

        maxv, maxw = 0.3, 0.3
        if abs(linear_vel) <= maxv:
            if abs(angular_vel) <= maxw:
                lv_lim, av_lim = linear_vel, angular_vel
            else:
                rd     = linear_vel / angular_vel
                lv_lim = maxw * np.sign(linear_vel) * abs(rd)
                av_lim = maxw * np.sign(angular_vel)
        else:
            if abs(angular_vel) <= 0.001:
                lv_lim, av_lim = maxv * np.sign(linear_vel), 0.0
            else:
                rd = linear_vel / angular_vel
                if abs(rd) >= maxv / maxw:
                    lv_lim = maxv * np.sign(linear_vel)
                    av_lim = maxv * np.sign(angular_vel) / abs(rd)
                else:
                    lv_lim = maxw * np.sign(linear_vel) * abs(rd)
                    av_lim = maxw * np.sign(angular_vel)

        self._save_visualization(
            current_image_PIL, self.goal_image_PIL,
            goal_pose_torch[0], waypoints[0],
            lv_lim, av_lim, metric_waypoint_spacing,
            mask_number.numpy() if hasattr(mask_number, 'numpy') else np.array([mid])
        )

        print("linear angular", lv_lim, av_lim)
        return lv_lim, av_lim

    def _save_visualization(self, cur_img, goal_img, goal_pose, waypoints,
                             linear_vel, angular_vel, spacing, mask_number):
        fig = plt.figure(figsize=(34, 16), dpi=80)
        gs  = fig.add_gridspec(2, 2)
        ax_ob   = fig.add_subplot(gs[0, 0])
        ax_goal = fig.add_subplot(gs[1, 0])
        ax_pos  = fig.add_subplot(gs[:, 1])

        ax_ob.imshow(np.array(cur_img).astype(np.uint8))
        ax_goal.imshow(np.array(goal_img).astype(np.uint8))

        x_seq     = waypoints[:, 0]
        y_seq_inv = -waypoints[:, 1]
        ax_pos.plot(
            np.insert(y_seq_inv, 0, 0.0),
            np.insert(x_seq,     0, 0.0),
            linewidth=4.0, markersize=12, marker='o', color='blue'
        )

        mask_texts = [
            "satellite only", "pose and satellite", "satellite and image", "all",
            "pose only", "pose and image", "image only", "language only", "language and pose"
        ]
        mask_type = int(mask_number[0])
        if mask_type < len(mask_texts):
            ax_pos.annotate(mask_texts[mask_type], xy=(1.0, 0.0),
                            xytext=(-20, 20), fontsize=18, textcoords='offset points')

        ax_ob.set_title("Egocentric current image", fontsize=18)
        ax_goal.set_title("Egocentric goal image",  fontsize=18)
        ax_pos.set_xlim(-3.0, 3.0)
        ax_pos.set_ylim(-0.1, 10.0)
        ax_pos.set_title("Normalized generated 2D trajectories from OmniVLA", fontsize=18)
        ax_pos.tick_params(axis='x', labelsize=15)
        ax_pos.tick_params(axis='y', labelsize=15)

        os.makedirs(self.datastore_path_image, exist_ok=True)
        save_path = os.path.join(
            self.datastore_path_image, f"{self.count_id}_ex_omnivla_edge.jpg"
        )
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Trayectoria guardada en: {save_path}")


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    print("="*60)
    print("  OmniVLA-edge  —  Inferencia CPU")
    print("="*60)

    # ── Modalidad ────────────────────────────────────────────────
    # Cambia estas 4 variables para elegir el tipo de goal:
    pose_goal  = False   # True = usar GPS
    satellite  = False   # True = usar mapa satelital
    image_goal = False   # True = usar imagen goal
    lan_prompt = True    # True = usar texto

    imgsize      = (96,  96)
    imgsize_clip = (224, 224)

    # ── Goal ─────────────────────────────────────────────────────
    lan_inst_prompt = "blue trash bin"

    goal_lat, goal_lon, goal_compass_deg = 37.8738930785863, -122.26746181032362, 0.0
    goal_utm     = utm.from_latlon(goal_lat, goal_lon)
    goal_compass = -float(goal_compass_deg) / 180.0 * math.pi

    goal_image_PIL = Image.open("./inference/goal_img.jpg").convert("RGB").resize(imgsize)

    # ── Parámetros del modelo ────────────────────────────────────
    model_params = {
        "model_type":               "omnivla-edge",
        "len_traj_pred":            8,
        "learn_angle":              True,
        "context_size":             5,
        "obs_encoder":              "efficientnet-b0",
        "encoding_size":            256,
        "obs_encoding_size":        1024,
        "goal_encoding_size":       1024,
        "late_fusion":              False,
        "mha_num_attention_heads":  4,
        "mha_num_attention_layers": 4,
        "mha_ff_dim_factor":        4,
        "clip_type":                "ViT-B/32",
    }

    device = torch.device("cpu")
    print(f"\nUsing device: cpu")

    # ── Cargar CLIP text encoder (siempre en PyTorch) ────────────
    print("Cargando CLIP text encoder...")
    _, text_encoder, _ = load_model(
        os.path.join(MODEL_WEIGHTS_PATH, "omnivla-edge.pth"),
        model_params, device
    )
    text_encoder = text_encoder.to(device).eval()

    # ── Inicializar backend de inferencia ────────────────────────
    print("\nInicializando backend de inferencia...")
    backend = InferenceBackend(model_params, device)

    # ── Correr inferencia ────────────────────────────────────────
    print("\nEjecutando inferencia...\n")
    inference = Inference(
        save_dir        = "./inference",
        lan_inst_prompt = lan_inst_prompt,
        goal_utm        = goal_utm,
        goal_compass    = goal_compass,
        goal_image_PIL  = goal_image_PIL,
    )
    inference.run()
