```
cmake \
  -S inference \
  -B build/inference \
  -DCMAKE_BUILD_TYPE=Debug \
  -DQNN_SDK_ROOT=/actual/path/to/qnn/sdk
```