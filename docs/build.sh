#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOC_LANG=$1

# make sure language is only en or zh
if [ "$DOC_LANG" != "en" ] && [ "$DOC_LANG" != "zh" ]; then
    echo "Language must be en or zh"
    exit 1
fi

cd $SCRIPT_DIR

# Set locale to avoid sphinx locale errors
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

SLIME_DOC_LANG=$DOC_LANG sphinx-build -b html -D language=$DOC_LANG --conf-dir ./  ./$DOC_LANG ./build/$DOC_LANG