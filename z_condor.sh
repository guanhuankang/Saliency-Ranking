executable = z_train.sh
requirements = (CUDADeviceName == "Tesla V100-SXM2-32GB")
request_GPUs = 4
error      = error.txt
log        = output.txt
output     = stdout
queue