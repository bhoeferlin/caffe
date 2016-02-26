REM if exist "../src/caffe/proto/caffe.pb.h" (
REM    echo caffe.pb.h remains the same as before
REM ) else (
    echo caffe.pb.h is being generated
    "%SHARED_PATH%/external/caffe/3rdparty/bin/protoc" -I="%SHARED_PATH%/external/caffe/src/caffe/proto" --cpp_out="%SHARED_PATH%/external/caffe/src/caffe/proto" "%SHARED_PATH%/external/caffe/src/caffe/proto/caffe.proto"
REM )

