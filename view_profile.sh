#!/bin/bash
# Assumes: 
# subdirectory profiling
# gprof2dot, dot, profile_eye

if [ $# = 0 ]
then
	profile=$(ls -tr profiling/*.pstats | tail -1)
	if [[ $profile != *"pstats" ]]
	then
		echo "Error: No profile found. Store as .pstats or pass argument"
		exit 1 
	fi
	echo "Viewing latest profile $profile"
else
	profile = $1
fi
profile_page=${profile%pstats}'html'
profile_image=${profile%pstats}'png'
gprof2dot -f pstats $profile | profile_eye --file-colon-line-colon-label-format > $profile_page
gprof2dot -f pstats $profile | dot -Tpng -o $profile_image
echo "Created image $profile_image and page $profile_page"
echo "Opening the image"
xdg-open $profile_image
