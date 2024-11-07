#foldername=$1; shift

set -e

python system/controller/topological/topological_navigation.py $@ --head 1
restores=$(ls logs/ | grep '^step' | sed 's/^step//g;s/.pkl$//g' | sort -h)
for i in $restores; do
	python system/controller/topological/topological_navigation.py $@ --restore $i
done
