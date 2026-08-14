```
cmake \
  -S inference \
  -B build/inference \
  -DCMAKE_BUILD_TYPE=Debug \
  -DQNN_SDK_ROOT=/actual/path/to/qnn/sdk
```

export LD_LIBRARY_PATH=/root/qairt-2.35/lib
export QNN_BACKEND_PATH=/root/qairt-2.35/lib/libQnnHtp.so