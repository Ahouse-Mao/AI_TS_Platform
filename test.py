import torch; print('CUDA可用:', torch.cuda.is_available())
print(f'当前GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无GPU"}')
print(torch.cuda.current_device())