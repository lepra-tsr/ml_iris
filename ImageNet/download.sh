#!/bin/sh

if [ $# -ne 2 ]; then
  echo "# call me like below:"
  echo "cat urllist.txt | xargs -P 5 -n 2 ./download.sh"
  exit 1
fi

# 末尾に改行が入ってるとエラーになる
file_name=img/${1%$'\r'}
url=${2%$'\r'} 

mkdir -p img
if [ -e "${file_name}" ]; then
  echo skip: "${file_name}" is already exists.
fi
if [ ! -e "${file_name}" ]; then
  # HTTPリクエストが失敗したら標準出力しない、進捗は非表示、curlのエラーは表示
  curl -X GET --retry 3 --fail -L "${url}" > "${file_name}"
fi
