#!/bin/bash

process=$1
out=$1'.txt'
pid=$(ps -e|grep $process| grep -v 'grep' | awk '{pring $1;}')
while true
do
    mem=`cat /proc/$pid/status|grep -e VmRSS`
    cpu=`top -n 1 -p $pid|tail -3|head -1|awk '{ssd=NF-4} {print $ssd}'`

    echo $(date+"%y-%m-%d-%H:%M:%S") $mem $cpu >> $out
    cat /proc/meminfo | grep -E 'MemTotal|MenFree|Cached' >> $out
    echo >> $out
    sleep 180
done