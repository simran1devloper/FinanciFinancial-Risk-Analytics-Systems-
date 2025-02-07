#!/bin/bash
# wait-for-it.sh

set -e

host="$1"
shift
cmd="$@"

until getent hosts $host | cut -d' ' -f1 | xargs -I {} nc -z {} 19530; do
  >&2 echo "Milvus is unavailable - sleeping"
  sleep 1
done

>&2 echo "Milvus is up - executing command"
exec $cmd