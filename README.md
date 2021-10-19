# Pytorch Implementation for BRECQ

The original repository is [here](https://https://github.com/yhhhli/BRECQ.git)

In this repository, BRECQ technique is applied to train MobileNetV2 model on the <b>MNIST</b> dataset

Repo used to train the MobileNetV2 is [here](https://github.com/Mayurji/Image-Classification-PyTorch.git)

### To test a pretrained MobileNetV2 model on 2 bit weights and 4 bit activation 
```
!python main_imagenet.py --data_path './' --arch mobilenetv2 --test_before_calibration --num_samples 1024 --n_bits_w 2 --n_bits_a 4 --act_quant
```

Results:

                                                                       (Without BRECQ)   (Weight Only)      (Full) 
                                                                         Quantization    Quantization    Quantization
|   Model   | Precision | Hyper-Params                                   | Accuracy |    | Accuracy |    | Accuracy |
| :-------: | --------- | ---------------------------------------------- | -------- |    | -------- |    | -------- |
| MobileNetV2 | W8A8     | --num_samples 1024 --n_bits_w 8 --n_bits_a 8  |  97.93   |    |  97.86   |    |  97.919  |
| MobileNetV2 | W4A8     | --num_samples 1024 --n_bits_w 4 --n_bits_a 8  |  95.5    |    |  97.919  |    |  97.909  |
| MobileNetV2 | W2A8     | --num_samples 1024 --n_bits_w 2 --n_bits_a 8  |  15.139  |    |  96.339  |    |  96.379  |
| MobileNetV2 | W4A4     | --num_samples 1024 --n_bits_w 4 --n_bits_a 4  |  95.5    |    |  97.919  |    |  96.479  |
| MobileNetV2 | W2A4     | --num_samples 1024 --n_bits_w 2 --n_bits_a 4  |  15.139  |    |  96.339  |    |  94.649  |
| MobileNetV2 | W3A3     | --num_samples 1024 --n_bits_w 3 --n_bits_a 3  |  78.0    |    |  97.569  |    |  83.489  |
| MobileNetV2 | W2A2     | --num_samples 1024 --n_bits_w 2 --n_bits_a 2  |  15.139  |    |  96.339  |    |  18.969  |


