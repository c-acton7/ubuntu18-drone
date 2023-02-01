# SampleSolution

Run this command in shell every time before using OpenVINO to create environment.

```shell
source /opt/intel/openvino_2021/bin/setupvars.sh
```

Or edit .bashrc

```shell
vi ~/.bashrc
```

Add this line to the end of the file

```shell
source /opt/intel/openvino_2021/bin/setupvars.sh
```

## Convert Model (already converted so no need to do this)

1. Export ONNX model

   ```shell
   python ./tools/export_onnx.py --cfg_path ${CONFIG_PATH} --model_path ${PYTORCH_MODEL_PATH}
   ```

2. Use *onnx-simplifier* to simplify it

   ``` shell
   python -m onnxsim ${INPUT_ONNX_MODEL} ${OUTPUT_ONNX_MODEL}
   ```

3. Convert to OpenVINO

   ``` shell
   cd <INSTSLL_DIR>/openvino_2021/deployment_tools/model_optimizer
   ```

   Install requirements for convert tool

   ```shell
   sudo ./install_prerequisites/install_prerequisites_onnx.sh
   ```

   Then convert model. Notice: mean_values and scale_values should be the same with your training settings in YAML config file.
   ```shell
   python3 mo_onnx.py --input_model <ONNX_MODEL> --mean_values [103.53,116.28,123.675] --scale_values [57.375,57.12,58.395]
   ```

## Build


```shell
cd autodrone-openvino
mkdir build
cd build
cmake ..
make
mkdir FP32
```

## Run demo

! First, move nanodet openvino model files (.bin .xml .mapping) to the FP32 folder. Then run these commands:

### Webcam

```shell
nanodet_demo 0 0
```

### Inference images

```shell
nanodet_demo 1 IMAGE_FOLDER/*.jpg
```

### Inference video

```shell
nanodet_demo 2 VIDEO_PATH
```

### Benchmark

```shell
nanodet_demo 3 0
```
