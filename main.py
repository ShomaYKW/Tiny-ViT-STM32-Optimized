import torch 
import torch.nn as nn
import os

from functools import partial
import torch
import torch.nn as nn
from timm.models.vision_transformer import DropPath, Mlp, Attention as BaseAttn

from model.pretrained_light import VisionTransformer
from conversion.conversion import conversion_pipeline
from evaluate.evaluate import get_model_info
#from model.pretrained_light import 

def vit_pico_patch32(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=32,       
        embed_dim=96,        
        depth=4,          
        num_heads=3 ,        
        mlp_ratio=2,
        qkv_bias=True,  
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model 

def vit_pico_patch16(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,       
        embed_dim=64,        
        depth=4,          
        num_heads=4,        
        mlp_ratio=2,
        qkv_bias=True,  
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model 


def vit_nano_patch16(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=64,      
        depth=6,            
        num_heads=2,        
        mlp_ratio=2,        
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model 


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_shape = [1,3,224,224]

    """
    vit_pico_patch_32 = vit_pico_patch32(pretrained=False)
    get_model_info(vit_pico_patch_32, input_shape)
    ONNX_PATH = "checkpoints/vit_pico_patch_32.onnx"
    TF_PATH = "checkpoints/vit_pico_patch_32"
    TFLITE_PATH = "saved_model/vit_pico_patch_32.tflite"
    conversion_pipeline(ONNX_PATH, TF_PATH, TFLITE_PATH, vit_pico_patch_32 , input_shape, device)
    """

    vit_pico_patch_16_P2 = vit_pico_patch16(pretrained=False)
    get_model_info(vit_pico_patch_16_P2, input_shape)
    ONNX_PATH = "checkpoints/vit_pico_patch_16_P2.onnx"
    TF_PATH = "checkpoints/vit_pico_patch_16_P2"
    TFLITE_PATH = "saved_model/vit_pico_patch_16_P2.tflite"
    conversion_pipeline(ONNX_PATH, TF_PATH, TFLITE_PATH, vit_pico_patch_16_P2 , input_shape, device)

    vit_nano_patch_16 = vit_nano_patch16(pretrained=False)
    get_model_info(vit_nano_patch_16, input_shape)
    ONNX_PATH = "checkpoints/vit_nano_patch_16.onnx"
    TF_PATH = "checkpoints/vit_nano_patch_16"
    TFLITE_PATH = "saved_model/vit_nano_patch_16.tflite"
    conversion_pipeline(ONNX_PATH, TF_PATH, TFLITE_PATH, vit_nano_patch_16 , input_shape, device)




