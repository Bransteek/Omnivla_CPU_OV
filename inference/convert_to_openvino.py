"""
convert_to_openvino.py
──────────────────────
Convierte el modelo OmniVLA-edge a formato OpenVINO IR con cuantización INT8.
Esto reduce el tamaño del modelo a la mitad y acelera la inferencia en CPU Intel.

Ejecutar UNA SOLA VEZ antes de usar run_omnivla_edge_cpu.py:
    python inference/convert_to_openvino.py
"""

import os
import sys
import torch
import numpy as np
sys.path.insert(0, '.')

# ── Rutas ────────────────────────────────────────────────────────
MODEL_PATH    = "./omnivla-edge/omnivla-edge.pth"
OUTPUT_DIR    = "./omnivla-edge-ov"
OUTPUT_MODEL  = os.path.join(OUTPUT_DIR, "omnivla_edge.xml")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("="*60)
    print("  Convirtiendo OmniVLA-edge a OpenVINO IR")
    print("="*60)

    # ── Verificar que existe el modelo ───────────────────────────
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}\n"
            "Asegúrate de haber ejecutado: git clone https://huggingface.co/NHirose/omnivla-edge"
        )

    # ── Cargar el modelo ─────────────────────────────────────────
    print("\n[1/4] Cargando modelo PyTorch...")
    from utils_policy import load_model

    model_params = {
        "model_type":          "omnivla-edge",
        "len_traj_pred":       8,
        "learn_angle":         True,
        "context_size":        5,
        "obs_encoder":         "efficientnet-b0",
        "encoding_size":       256,
        "obs_encoding_size":   1024,
        "goal_encoding_size":  1024,
        "late_fusion":         False,
        "mha_num_attention_heads":  4,
        "mha_num_attention_layers": 4,
        "mha_ff_dim_factor":        4,
        "clip_type":           "ViT-B/32",
    }

    device = torch.device("cpu")
    model, text_encoder, _ = load_model(MODEL_PATH, model_params, device)
    model = model.to(device).eval()
    print("   Modelo cargado correctamente.")

    # ── Crear inputs de ejemplo para tracing ────────────────────
    print("\n[2/4] Creando inputs de ejemplo...")
    batch_size   = 1
    context_size = 6   # 6 frames de historia

    # Dimensiones que usa el modelo internamente
    obs_images    = torch.zeros(batch_size, context_size * 3, 96, 96)   # 6 frames RGB
    goal_pose     = torch.zeros(batch_size, 4)                           # x, y, cos, sin
    map_images    = torch.zeros(batch_size, 9, 96, 96)                   # 3 imágenes de mapa
    goal_image    = torch.zeros(batch_size, 3, 96, 96)                   # imagen goal
    modality_id   = torch.tensor([7])                                     # language only
    feat_text     = torch.zeros(batch_size, 512)                          # embedding CLIP texto
    cur_large_img = torch.zeros(batch_size, 3, 224, 224)                  # imagen grande actual

    example_inputs = (
        obs_images, goal_pose, map_images,
        goal_image, modality_id, feat_text, cur_large_img
    )
    print("   Inputs creados.")

    # ── Exportar a ONNX (paso intermedio) ───────────────────────
    print("\n[3/4] Exportando a ONNX...")
    onnx_path = os.path.join(OUTPUT_DIR, "omnivla_edge.onnx")

    try:
        torch.onnx.export(
            model,
            example_inputs,
            onnx_path,
            opset_version=14,
            input_names=[
                "obs_images", "goal_pose", "map_images",
                "goal_image", "modality_id", "feat_text", "cur_large_img"
            ],
            output_names=["predicted_actions", "distances", "mask_number"],
            dynamic_axes={
                "obs_images":    {0: "batch"},
                "goal_pose":     {0: "batch"},
                "map_images":    {0: "batch"},
                "goal_image":    {0: "batch"},
                "modality_id":   {0: "batch"},
                "feat_text":     {0: "batch"},
                "cur_large_img": {0: "batch"},
            },
        )
        print(f"   ONNX guardado en: {onnx_path}")
    except Exception as e:
        print(f"   Error exportando ONNX: {e}")
        print("   Intentando con torch.jit.trace como alternativa...")
        _fallback_torchscript(model, example_inputs, OUTPUT_DIR)
        return

    # ── Convertir ONNX → OpenVINO IR ────────────────────────────
    print("\n[4/4] Convirtiendo ONNX a OpenVINO IR con cuantización INT8...")
    try:
        import openvino as ov
        from openvino.tools.mo import convert_model

        ov_model = ov.convert_model(onnx_path)

        # Cuantización INT8 con NNCF para reducir memoria y acelerar CPU
        try:
            import nncf
            import nncf.torch

            print("   Aplicando cuantización INT8 con NNCF...")

            def _calibration_data():
                """Genera datos de calibración sintéticos."""
                for _ in range(10):
                    yield {
                        "obs_images":    np.random.rand(1, 18, 96, 96).astype(np.float32),
                        "goal_pose":     np.random.rand(1, 4).astype(np.float32),
                        "map_images":    np.random.rand(1, 9, 96, 96).astype(np.float32),
                        "goal_image":    np.random.rand(1, 3, 96, 96).astype(np.float32),
                        "modality_id":   np.array([7], dtype=np.int64),
                        "feat_text":     np.random.rand(1, 512).astype(np.float32),
                        "cur_large_img": np.random.rand(1, 3, 224, 224).astype(np.float32),
                    }

            quantized_model = nncf.quantize(
                ov_model,
                calibration_dataset=nncf.Dataset(list(_calibration_data())),
                preset=nncf.QuantizationPreset.MIXED,
            )
            ov.save_model(quantized_model, OUTPUT_MODEL)
            print(f"   Modelo INT8 guardado en: {OUTPUT_MODEL}")

        except Exception as eq:
            print(f"   NNCF no disponible ({eq}), guardando FP32...")
            ov.save_model(ov_model, OUTPUT_MODEL)
            print(f"   Modelo FP32 guardado en: {OUTPUT_MODEL}")

    except Exception as e:
        print(f"   Error convirtiendo a OpenVINO: {e}")
        print("   El modelo ONNX está disponible para uso directo con onnxruntime.")

    print("\n" + "="*60)
    print("  Conversión completada.")
    print(f"  Modelo OpenVINO en: {OUTPUT_DIR}/")
    print("  Ahora puedes usar: python inference/run_omnivla_edge_cpu.py")
    print("="*60)


def _fallback_torchscript(model, inputs, output_dir):
    """Fallback: guarda como TorchScript si ONNX falla."""
    try:
        traced = torch.jit.trace(model, inputs)
        path = os.path.join(output_dir, "omnivla_edge_traced.pt")
        traced.save(path)
        print(f"   TorchScript guardado en: {path}")
    except Exception as e:
        print(f"   TorchScript también falló: {e}")


if __name__ == "__main__":
    main()
