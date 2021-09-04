#!/bin/sh
echo "["
for i in `find eval_results/ -type f -iname *eval.log`; do
  ev_type=$(echo $i | awk -F '/' '{print $6}' | sed 's/_eval_output//')
  #echo $ev_type
  model=$(echo $i | awk -F '/' '{print $2}')
  #echo $model
  #echo $i | awk -F '/' '{print $3}'
  beam=$(echo $i | awk -F '/' '{print $4}' | sed 's/beam_//')
  #echo $beam
  f1=$(cat $i | grep 'Official avg F1' | head -n 1 | sed 's/Official avg F1: //')
  if [ -z "$f1" ]; then
    f1="-1"
  fi
  #echo $f1
  #echo ''
  echo "{\"eval_type\" : \"$ev_type\", \"f1\" : $f1, \"model\" : \"$model\", \"beam_size\" : $beam},"
done
echo "]"
