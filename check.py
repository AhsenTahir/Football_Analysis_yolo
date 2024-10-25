import torch

def check_pytorch_gpu():
    if torch.cuda.is_available():
        print("CUDA is available. GPU can be used.")
        print(f"Current GPU device: {torch.cuda.get_device_name(0)}")
        
        # Check if PyTorch is actually using CUDA
        x = torch.rand(1).cuda()
        if x.is_cuda:
            print("PyTorch is using GPU.")
        else:
            print("PyTorch is not using GPU, despite CUDA being available.")
    else:
        print("CUDA is not available. GPU cannot be used.")
        print("PyTorch will use CPU.")

if __name__ == "__main__":
    check_pytorch_gpu()

