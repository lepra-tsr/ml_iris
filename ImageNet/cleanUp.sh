#!/bin/sh

for f in `ls img/*.jpg`
do
  type_result=`file $f`
  if [[ $type_result =~ .*JPEG.* ]];then
    : # this is JPEG. do NOTHING
  else
    echo remove $type_result
    rm -f $f
  fi
done