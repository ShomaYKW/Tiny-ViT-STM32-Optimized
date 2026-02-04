import onnx
import torch


#check the conversion, but the model must be in .onnx format
def check_onnx(path):

    model = onnx.load(path)
    
    # Check that the IR is well formed
    onnx.checker.check_model(model)
    
    onnx.printer.to_text(model)

    return model

