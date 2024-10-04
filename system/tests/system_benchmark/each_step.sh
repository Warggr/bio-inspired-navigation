#foldername=$1; shift

set -e

python system/tests/system_benchmark/long_unknown_nav.py $@ --head 1
restores=$(ls logs/ | grep '^step' | sed 's/^step//g;s/.pkl$//g' | sort -h)
for i in $restores; do
	python system/tests/system_benchmark/long_unknown_nav.py $@ --restore $i
done
