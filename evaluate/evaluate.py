import torch
import torch.nn as nn
import time
import os


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def count_conv_parameters(model):
    conv_params = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv_params += sum(p.numel() for p in module.parameters())
    return conv_params


def get_model_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def get_layer_wise_params(model):
    layer_params = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params = sum(p.numel() for p in module.parameters())
            layer_params[name] = params
    return layer_params


def get_conv_layer_info(model):
    conv_info = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_info[name] = {
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'parameters': sum(p.numel() for p in module.parameters())
            }
    return conv_info


def measure_inference_time(model, input_size, device='cpu', num_runs=100):
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  
    return avg_time


def calculate_flops_conv2d(module, input_size):
    batch_size, in_channels, in_h, in_w = input_size
    out_channels = module.out_channels
    kernel_h, kernel_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
    
    # Output dimensions
    out_h = (in_h + 2 * module.padding[0] - kernel_h) // module.stride[0] + 1
    out_w = (in_w + 2 * module.padding[1] - kernel_w) // module.stride[1] + 1

    flops = batch_size * out_channels * out_h * out_w * (in_channels * kernel_h * kernel_w + (1 if module.bias is not None else 0))
    
    return flops, (batch_size, out_channels, out_h, out_w)


def estimate_model_flops(model, input_size):
    model.eval()
    total_flops = 0
    
    def hook_fn(module, input, output):
        nonlocal total_flops
        if isinstance(module, nn.Conv2d):
            flops, _ = calculate_flops_conv2d(module, input[0].shape)
            total_flops += flops
        elif isinstance(module, nn.Linear):
            total_flops += module.in_features * module.out_features
    
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        dummy_input = torch.randn(input_size)
        model(dummy_input)
    
    for hook in hooks:
        hook.remove()
    
    return total_flops


def save_model_to_disk(model, filepath='model_temp.pth'):
    torch.save(model.state_dict(), filepath)
    size_mb = os.path.getsize(filepath) / (1024 ** 2)
    os.remove(filepath)  
    return size_mb


def compare_models(original_model, pruned_model, input_size):

    
    # 1. Parameter Count
    print("\n1. parameter count")
    print("-" * 70)
    orig_total, orig_trainable = count_parameters(original_model)
    pruned_total, pruned_trainable = count_parameters(pruned_model)
    
    print(f"Original Model:")
    print(f"  Total parameters:      {orig_total:,}")
    print(f"  Trainable parameters:  {orig_trainable:,}")
    
    print(f"\nPruned Model:")
    print(f"  Total parameters:      {pruned_total:,}")
    print(f"  Trainable parameters:  {pruned_trainable:,}")
    
    reduction = (1 - pruned_total / orig_total) * 100
    print(f"\n✓ Parameter Reduction:   {reduction:.2f}%")
    print(f"✓ Compression Ratio:     {orig_total / pruned_total:.2f}x")
    
    # 2. Conv Layer Parameters
    print("\n2. Conv layer parameter count")
    print("-" * 70)
    orig_conv = count_conv_parameters(original_model)
    pruned_conv = count_conv_parameters(pruned_model)
    
    print(f"Original Conv Params:    {orig_conv:,}")
    print(f"Pruned Conv Params:      {pruned_conv:,}")
    conv_reduction = (1 - pruned_conv / orig_conv) * 100
    print(f"✓ Conv Param Reduction:  {conv_reduction:.2f}%")
    
    # 3. Model Size
    print("\n3. momery size")
    print("-" * 70)
    orig_size = get_model_size_mb(original_model)
    pruned_size = get_model_size_mb(pruned_model)
    
    print(f"Original Model Size:     {orig_size:.2f} MB")
    print(f"Pruned Model Size:       {pruned_size:.2f} MB")
    size_reduction = (1 - pruned_size / orig_size) * 100
    print(f"✓ Size Reduction:        {size_reduction:.2f}%")
    
    # 4. Disk Size
    print("\n4. disk size")
    print("-" * 70)
    orig_disk = save_model_to_disk(original_model, 'orig_temp.pth')
    pruned_disk = save_model_to_disk(pruned_model, 'pruned_temp.pth')
    
    print(f"Original Disk Size:      {orig_disk:.2f} MB")
    print(f"Pruned Disk Size:        {pruned_disk:.2f} MB")
    disk_reduction = (1 - pruned_disk / orig_disk) * 100
    print(f"✓ Disk Size Reduction:   {disk_reduction:.2f}%")
    
    # 5. Layer-wise Comparison
    print("\n5. Conv layer - wise parameter conpare")
    print("-" * 70)
    orig_conv_info = get_conv_layer_info(original_model)
    pruned_conv_info = get_conv_layer_info(pruned_model)
    
    print(f"{'Layer Name':<30} {'Original Channels':<20} {'Pruned Channels':<20} {'Reduction %'}")
    print("-" * 70)
    
    for name in orig_conv_info:
        if name in pruned_conv_info:
            orig_out = orig_conv_info[name]['out_channels']
            pruned_out = pruned_conv_info[name]['out_channels']
            reduction_pct = (1 - pruned_out / orig_out) * 100
            print(f"{name:<30} {orig_out:<20} {pruned_out:<20} {reduction_pct:.1f}%")
    
    # 6. Inference Time
    print("\n6. inference")
    print("-" * 70)
    orig_time = measure_inference_time(original_model, input_size)
    pruned_time = measure_inference_time(pruned_model, input_size)
    
    print(f"Original Model:          {orig_time:.2f} ms")
    print(f"Pruned Model:            {pruned_time:.2f} ms")
    speedup = orig_time / pruned_time
    print(f"Speedup:               {speedup:.2f}x")
    
    # 7. FLOPs Estimation
    print("\n7.  (FLOPs)")
    print("-" * 70)
    orig_flops = estimate_model_flops(original_model, input_size)
    pruned_flops = estimate_model_flops(pruned_model, input_size)
    
    print(f"Original Model FLOPs:    {orig_flops:,}")
    print(f"Pruned Model FLOPs:      {pruned_flops:,}")
    flops_reduction = (1 - pruned_flops / orig_flops) * 100
    print(f"FLOPs Reduction:       {flops_reduction:.2f}%")
    
    # summary
    print("Ssummary")
    print(f" Parameters reduced by:     {reduction:.2f}%")
    print(f" Model size reduced by:     {size_reduction:.2f}%")
    print(f" Inference speedup:         {speedup:.2f}x")
    print(f" FLOPs reduced by:          {flops_reduction:.2f}%")
    print("="*70)
    
    return {
        'param_reduction': reduction,
        'size_reduction': size_reduction,
        'speedup': speedup,
        'flops_reduction': flops_reduction,
        'compression_ratio': orig_total / pruned_total
    }


def get_model_info(original_model, input_size):

    
    # 1. Parameter Count
    print("\n1. parameter count")
    orig_total, orig_trainable = count_parameters(original_model)
    print(f"  Total parameters:      {orig_total:,}")
    print(f"  Trainable parameters:  {orig_trainable:,}")
    
    # 2. Conv Layer Parameters
    print("\n2. Conv layer parameter count")
    orig_conv = count_conv_parameters(original_model)
    print(f" Conv Params:    {orig_conv:,}")
    
    # 3. Model Size
    print("\n3. momery size")
    orig_size = get_model_size_mb(original_model)
    print(f" Model Size:     {orig_size:.2f} MB")
    
    # 4. Disk Size
    print("\n4. disk size")
    orig_disk = save_model_to_disk(original_model, 'orig_temp.pth')
    print(f"Disk Size:      {orig_disk:.2f} MB")
    
    # 6. Inference Time
    print("\n6. inference")
    orig_time = measure_inference_time(original_model, input_size)
    print(f"Original Model:          {orig_time:.2f} ms")

    # 7. FLOPs Estimation
    print("\n7.  (FLOPs)")
    orig_flops = estimate_model_flops(original_model, input_size)
    print(f"Original Model FLOPs:    {orig_flops:,}")
