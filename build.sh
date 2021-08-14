export PATH=~/bin:$PATH
poj=$1
bazel build //main:$poj

echo "============= build finished ============"
if [ -e bazel-bin/main/$poj ];then
    ./bazel-bin/main/$poj
fi