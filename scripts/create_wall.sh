x0=$1
x1=$2
y0=$3
y1=$4
name=${5:-wall}

cat system/controller/simulation/environment/Savinov_val3/walls/template.obj.mustache | sed "
s/{{ x0 }}/$x0/
s/{{ x1 }}/$x1/
s/{{ y0 }}/$y0/
s/{{ y1 }}/$y1/
s/{{ name }}/$name/
"
