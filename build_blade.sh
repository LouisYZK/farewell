export PATH=~/bin:$PATH
poj=$1
blade build //main:$poj

echo "============= build finished ============"
if [ -e blade-bin/main/$poj ];then
    ./blade-bin/main/$poj
fi