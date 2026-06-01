#!/bin/bash
# 啟動 Qdrant（資料存於 src_2/qdrant_storage/）

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
STORAGE_DIR="$SCRIPT_DIR/qdrant_storage"

mkdir -p "$STORAGE_DIR"

# 若已存在同名 container 先移除
if docker ps -a --format '{{.Names}}' | grep -q '^qdrant_v2$'; then
    echo "Removing existing qdrant_v2 container..."
    docker rm -f qdrant_v2
fi

docker run -d \
    -p 6333:6333 \
    -v "$STORAGE_DIR":/qdrant/storage \
    --name qdrant_v2 \
    qdrant/qdrant

echo "Qdrant started → http://localhost:6333"
echo "Storage: $STORAGE_DIR"
