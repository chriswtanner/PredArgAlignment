#!/bin/bash
# python Test.py /data/people/christanner/ 42 false false cpu lstm 100 5 30 5 1.0
# $dn $sm $rc $dev lstm $hs$ns $ne$bs $lr

cd /home/christanner/researchcode/PredArgAlignment/src/
path=/data/people/christanner/
dev=cpu
queue=long

hs=(150) # 150 300 450)
num_steps=(3) # 3 10
num_epochs=(20) # 20  40
learning_rate=(1.0) 
batch_size=(5) # 5, 10
dirs=(3) # (25 26 32 35 38 40 42)
stitchMentions=(false) # false true
reverseCorpus=(false) # false true
ecbPlus=(false true) # false true

for sm in "${stitchMentions[@]}"
do
    for rc in "${reverseCorpus[@]}"
    do
	for hs in "${hs[@]}"
	do
	    for ns in "${num_steps[@]}"
	    do
		for ne in "${num_epochs[@]}"
		do
		    for lr in "${learning_rate[@]}"
		    do
			for bs in "${batch_size[@]}"
			do
			    for ep in "${ecbPlus[@]}"
			    do
				for dn in `seq 1 45`; # was 23 45
				# for dn in "${dirs[@]}"
				do
				    if [[ $dn -ne 15 && $dn -ne 17 ]]; then
					echo $dn $ep $sm $rc lstm $hs $ns $ne $bs $lr	    
					base=lstm_dir${dn}_${ep}_${stitchMentions}_${reverseCorpus}_h${hs}_ns${ns}_ne${ne}_bs${bs}_lr${lr}
					file=$base.csv
					qsub -l vf=5G -o $base.out -l ${queue} runLSTM_1b.sh $path $dn $ep $sm $rc $dev lstm $hs $ns $ne $bs $lr
					# qsub -l gpus=1 -l vf=5G -o $base.out -l ${queue} runLSTM_1b.sh $path $dn $ep $sm $rc $dev lstm $hs $ns $ne $bs $lr
				    fi
				done
			    done
			done
		    done
		done
	    done
	done
    done
done


