function dquine { echo '#' $@; $@; }

function archive {
        DATE=$(ls $1 --time-style "+%y%m%d" -l | cut -d ' ' -f 6)
        mv -i -v $1 ${1%.log}.$DATE.log
}

function extractNotebookImage {
        cat $1 | jq -r \
                ".cells[] | select(.source | join(\"\") | contains(\"$2\")) | .outputs[] | select(.output_type == \"display_data\") | .data[\"image/png\"]" \
               | while read -r line; do
                        echo "fig$((++i))"
                        echo "$line" | base64 -d > fig${i}.png
                done
}
