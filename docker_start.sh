#!/usr/bin/env bash
if [ $# -ne 2 ]; then
    echo "Usage: bash $0 <image-id-or-tag> <container-name>" >&2
    exit 1
fi
IMAGES_ID=$1
NAME=$2


docker run -u root -it -d --name "${NAME}" --net=host --pid=host --shm-size=500g --privileged=true \
    --device=/dev/davinci_manager --device=/dev/hisi_hdc --device=/dev/ummu --device=/dev/uburma \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    -v /root/host:/root/host  \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /var/log/npu/:/usr/slog \
    -v /mnt:/mnt \
    -v /data:/data \
    -v /usr/lib64:/usr/lib64   \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /home/:/home/ \
    -v /etc/hixlep:/etc/hixlep \
    -v /lib/route.conf:/lib/route.conf:ro \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -w /home \
    "${IMAGES_ID}" \
    bash
