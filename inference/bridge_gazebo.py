import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch
import numpy as np
import math
import time
from PIL import Image as PILImage
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.run_omnivla_edge_cpu import InferenceBackend
from inference.utils_policy import transform_images_map, load_model, transform_images_PIL_mask
import clip

class OmniVLAGazeboBridge(Node):
    def __init__(self):
        super().__init__('omnivla_bridge')
        
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
        self.device = torch.device("cpu")
        self.get_logger().info("Cargando modelo OmniVLA...")
        self.backend = InferenceBackend(model_params, self.device)
        
        _, self.text_encoder, _ = load_model(os.path.join("./omnivla-edge", "omnivla-edge.pth"), model_params, self.device)
        self.text_encoder = self.text_encoder.to(self.device).eval()

        self.context_queue = [] 
        self.bridge = CvBridge()
        
        self.mask_96 = np.ones((96, 96, 3), dtype=np.float32)
        self.mask_224 = np.ones((224, 224, 3), dtype=np.float32)
        
        self.modality_id = torch.tensor([6])
        goal_img_raw = PILImage.open("inference/goal_img.png").convert("RGB").resize((96, 96))
        self.goal_image_tensor = transform_images_PIL_mask(goal_img_raw, self.mask_96)
        
        sat_cur = PILImage.new("RGB", (352, 352), color=(0, 0, 0))
        sat_goal = PILImage.new("RGB", (352, 352), color=(0, 0, 0))
        self.current_map = transform_images_map(sat_cur)
        self.goal_map = transform_images_map(sat_goal)

        obj_inst_lan = clip.tokenize("xxxx", truncate=True)
        self.feat_text_lan = self.text_encoder.encode_text(obj_inst_lan)
        
        self.goal_pose_torch = torch.from_numpy(np.array([
            1.0 / 0.1, -10.0 / 0.1, np.cos(-90.0 / 180.0 * 3.1415), np.sin(-90.0 / 180.0 * 3.1415)
        ])).unsqueeze(0).float()
        
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.prediction_done = False
        
        self.get_logger().info("Puente listo. Esperando juntar 6 imágenes iniciales...")

    def image_callback(self, msg):

        if self.prediction_done:
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            current_image_PIL = PILImage.fromarray(cv_img)
            
            img_96 = current_image_PIL.resize((96, 96))
            img_224 = current_image_PIL.resize((224, 224))
            
            self.context_queue.append(img_96)
            
            # Si aún no tenemos 6 fotos, no hacemos nada
            if len(self.context_queue) < 6:
                return
            
            # Mantenemos el tamaño en 6 por si acaso
            if len(self.context_queue) > 6:
                self.context_queue.pop(0)
                
            self.get_logger().info("🧠 Calculando ruta de 8 pasos...")
            self.prediction_done = True
            
            # Formatear datos
            obs_images = transform_images_PIL_mask(self.context_queue, self.mask_96)
            obs_images = torch.cat(torch.split(obs_images, 3, dim=1), dim=1)
            obs_image_cur = torch.split(obs_images, 3, dim=1)[-1]
            
            cur_large_img = transform_images_PIL_mask(img_224, self.mask_224)
            map_images = torch.cat((self.current_map, self.goal_map, obs_image_cur), axis=1)
            
            # --- INFERENCIA ---
            predicted_actions, _, _ = self.backend.run(
                obs_images, self.goal_pose_torch, map_images,
                self.goal_image_tensor, self.modality_id, self.feat_text_lan, cur_large_img
            )
            
            waypoints = predicted_actions.float().numpy()[0]
            
            self.get_logger().info("✅ Ruta calculada. Ejecutando...")
            
            # --- EJECUTAR LOS 8 PASOS ---
            for i, step in enumerate(waypoints):
                step_copy = step.copy()
                step_copy[:2] *= 0.1 
                dx, dy, hx, hy = step_copy
                
                EPS = 1e-8
                DT = 1 / 3 
                
                if np.abs(dx) < EPS and np.abs(dy) < EPS:
                    lv = 0.0
                    av = 1.0 * math.atan2(hy, hx) / DT
                elif np.abs(dx) < EPS:
                    lv = 0.0
                    av = 1.0 * np.sign(dy) * np.pi / (2 * DT)
                else:
                    lv = dx / DT
                    av = np.arctan(dy / dx) / DT
                    
                lv_lim = float(np.clip(lv, 0.0, 1.0))
                av_lim = float(np.clip(av, -1.0, 1.0))
                
                cmd = Twist()
                cmd.linear.x = lv_lim
                cmd.angular.z = av_lim
                self.pub.publish(cmd)
                
                time.sleep(DT)
            
            # --- FRENADO MOMENTÁNEO ---
            cmd_stop = Twist()
            self.pub.publish(cmd_stop)
            

        except Exception as e:
            
            self.get_logger().error(f"❌ Error en cálculo: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = OmniVLAGazeboBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()