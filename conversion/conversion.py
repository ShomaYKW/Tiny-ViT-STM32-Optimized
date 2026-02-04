import torch
import torch.nn as nn
import torch.onnx
import numpy as np
import tensorflow as tf
import onnx
import os
from onnx_tf.backend import prepare
import subprocess


def convert_to_onnx(model, path, input_shape, device):

    model.eval()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dummy_input = torch.randn(input_shape, device=device)
    
    model.to(device)
    
    torch.onnx.export(
        model, 
        dummy_input,
        path,
        export_params=True,
        opset_version=12,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes=None 
    )
    print("PyTorch -> ONNX conversion complete.")


def convert_to_onnx_2(model, path, device):
 
    model.eval()
    model.to(device)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if input_shape is None:
        try:
            # 1. Try to find the Patch Embedding layer (Common in ViTs/MAE)
            # This works for timm-style ViTs and the MAE-Lite model you are using
            if hasattr(model, 'patch_embed'):
                # MAE/ViT usually stores img_size in patch_embed
                H = W = model.patch_embed.img_size[0] if isinstance(model.patch_embed.img_size, tuple) else model.patch_embed.img_size
                C = model.patch_embed.proj.in_channels # Input channels (usually 3)
                input_shape = (1, C, H, W)
                print(f"Auto-detected ViT input shape: {input_shape}")
            
            # 2. Fallback for standard CNNs (First Conv2d layer)
            elif hasattr(model, 'features') and isinstance(model.features[0], torch.nn.Conv2d):
                # This is a guess for generic CNNs
                C = model.features[0].in_channels
                # We can't know H/W for sure in CNNs, defaulting to standard 224
                input_shape = (1, C, 224, 224) 
                print(f"Auto-detected CNN input channels: {C}. Defaulting to 224x224.")

            else:
                raise AttributeError("Could not find 'patch_embed' or standard feature block.")

        except AttributeError:
            print("Could not auto-detect shape. Defaulting to (1, 3, 224, 224).")
            input_shape = (1, 3, 224, 224)


    dummy_input = torch.randn(input_shape, device=device)
    
    print(f"Exporting with input shape: {input_shape}")
    
    torch.onnx.export(
        model, 
        dummy_input,
        path,
        export_params=True,
        opset_version=12,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes=None 
    )
    print(f"conversion done")





def convert_to_tf(onnx_path, tf_path):
   
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)

    tf_rep.export_graph(tf_path)
    print(" ONNX -> TensorFlow conversion complete.")

def convert_to_tf_2(onnx_path, tf_path):

    print(f" Converting ONNX to TF: {onnx_path} -> {tf_path}")
    
    if not os.path.exists(onnx_path):
        print(f" ONNX file not found ")
        return False

    cmd = [
        "onnx2tf", 
        "-i", onnx_path, 
        "-o", tf_path,
        "-osd"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(" Conversion Successful")
        return True
    except subprocess.CalledProcessError:
        print("Conversion Failed. Check logs above.")
        return False
    








def convert_to_tflite_int8(tf_path, tflite_path, calibration_loader):
   
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    
    # 2. Enable INT8 Optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    

    def representative_dataset_gen():
        # Iterate through 100 images from your train/test loader
        count = 0
        for images, _ in calibration_loader:
            for i in range(len(images)):
                img = images[i].unsqueeze(0).cpu().numpy()
                yield [img]
                
                count += 1
                if count >= 100: return

    converter.representative_dataset = representative_dataset_gen
    
    # 4. Convert
    tflite_quant_model = converter.convert()
  
    # 5. Save
    with open(tflite_path, 'wb') as f:
        f.write(tflite_quant_model)
    
    size_mb = len(tflite_quant_model) / (1024 * 1024)
    print(f"TFLite (INT8) saved. Size: {size_mb:.2f} MB")
    
    return tflite_quant_model


def convert_to_tflite_int8_2(tf_path, tflite_path, input_shape):
    
    # load the saved TF model 
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    
    # enable INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Create a Synthetic Representative Dataset
    # Since accuracy doesn't matter, this is sufficient to set the quantization ranges.
    def representative_dataset_gen():
        for _ in range(100):
            # Generate random float32 data in range [0, 1]
            # Ensure this shape matches what your TF model expects
            # TensorFlow models are usually NHWC (Batch, Height, Width, Channels)
            dummy_data = np.random.rand(*input_shape).astype(np.float32)
            yield [dummy_data]

    converter.representative_dataset = representative_dataset_gen
    
    # Convert
    print("Converting model ")
    try:
        tflite_quant_model = converter.convert()
    except Exception as e:
        print(f"Conversion failed: {e}")
        return None
  
    # Save
    with open(tflite_path, 'wb') as f:
        f.write(tflite_quant_model)
    
    size_mb = len(tflite_quant_model) / (1024 * 1024)
    print(f"TFLite (INT8) saved to {tflite_path}")
    print(f"  Size: {size_mb:.2f} MB")
    
    return tflite_quant_model

    









def conversion_pipeline(ONNX_PATH, TF_PATH, TFLITE_PATH, model, input_shape, device):

    convert_to_onnx(model, ONNX_PATH, input_shape, device)
    convert_to_tf_2(ONNX_PATH, TF_PATH)
    print("\nStarting INT8 Quantization")
    try:
        convert_to_tflite_int8_2(TF_PATH, TFLITE_PATH, input_shape)
        print("\nDeployment chain complete.")
    except Exception as e:
        print(f"\n[Error] Quantization failed: {e}")













