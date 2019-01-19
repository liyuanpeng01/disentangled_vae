#!/bin/sh

path=$1

if [ $# -lt 2 ]
then
    number=9
else
    number=$2
fi

dir=${path}disentangle_anim; [ ! -e $dir ] && mkdir -p $dir

for i in `seq 0 ${number}`
do
	png_name="${path}disentangle_img/check_z${i}_*.png";
	gif_name="${path}disentangle_anim/anim_z${i}.gif";
	convert $png_name $gif_name;
done
