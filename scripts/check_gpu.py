import torch
import platform

print('===== SYSTEM CHECK =====')
print('Python:', platform.python_version())
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('CUDA version used by PyTorch:', torch.version.cuda)
    print('GPU name:', torch.cuda.get_device_name(0))
    print('GPU count:', torch.cuda.device_count())

    x = torch.randn(2048, 2048).cuda()
    y = torch.matmul(x, x)
    print('GPU tensor test:', y.shape)
    print('GPU test status: PASSED')
else:
    print('GPU test status: FAILED - CUDA not available')
