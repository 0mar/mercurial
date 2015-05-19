if [ $# = 0 ]
then
	echo "Error: Provide script and interpreter arguments for python"
	exit 1
fi
prof_dir="profiling"
if [ ! -d "$prof_dir" ]
then
	mkdir $prof_dir
	echo "Created directory 'profiling'"
fi
echo "Name for profiling result file?"
read file_name
if [ -z "$file_name" ]
then
	file_name=profile-$(date +%d-%m_%H:%M)
fi
echo "Profiling results stored in '$file_name.pstats'"

echo -e "Executing profiling of 'python3 $@'.\nResults are stored and can be viewed directly with './view_profile.sh'"
python3 -m cProfile -o $prof_dir/$file_name".pstats" "$@"
