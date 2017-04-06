#!/bin/bash
# python Test.py /data/people/christanner/ 42 false false cpu lstm 100 5 30 5 1.0
# $dn $sm $rc $dev lstm $hs$ns $ne$bs $lr

cd /home/christanner/researchcode/PredArgAlignment/src/
path=/data/people/christanner/
queue=vlong

hiddens=(800) # 1201) # 400) # 100 400 
num_steps=(7) # 3
num_epochs=(10) # 20
learning_rate=(1.0) 
batch_size=(5) # 5, 10
stitchMentions=(false) # true) # false true
reverseCorpus=(false) # false true
ep=ecbplus

# nn params
windowSize=(0) # 6)
nnmethod=full #sub full # or full
opts=(rms) # adam gd or rms
hiddensA=(1500)
hiddensB=(1000)
keep_inputsA=(0.9)
keep_inputsB=(0.7)
keep_inputsC=(0.7)
num_epochsA=(100)
batch_size2=(5)
learning_rateA=(0.0001 0.001 0.01 0.05)
moms=(0.001 0.01 0.1 0.3) #0.1 0.9) #  0.1 0.9)
subsample=(2)
penalty=(2)
activation=(relu) # sigmoid
for mom in "${moms[@]}"
do
    for ws in "${windowSize[@]}"
    do
	for sm in "${stitchMentions[@]}"
	do
	    for rc in "${reverseCorpus[@]}"
	    do
		for hs in "${hiddens[@]}"
		do
		    for ns in "${num_steps[@]}"
		    do
			for ne in "${num_epochs[@]}"
			do
			    for lr in "${learning_rate[@]}"
			    do
				for bs in "${batch_size[@]}"
				do
				    for opt in "${opts[@]}"
				    do
					for h1 in "${hiddensA[@]}"
					do
					    for h2 in "${hiddensB[@]}"
					    do
						for k0 in "${keep_inputsA[@]}"
						do
						    for k1 in "${keep_inputsB[@]}"
						    do
							for k2 in "${keep_inputsC[@]}"
							do
							    for ne2 in "${num_epochsA[@]}"
							    do
								for lr2 in "${learning_rateA[@]}"
								do
								    for bs2 in "${batch_size2[@]}"
								    do
									for sub in "${subsample[@]}"
									do
									    
									    for pen in "${penalty[@]}"
									    do
										
										for act in "${activation[@]}"
										do
										    echo $ws $sm $rc lstm $hs $ns $ne $lr $bs $nnmethod $opt $h1 $h2 $k0 $k1 $k2 $ne2 $lr2 $mom $sub $pen $act
										    base=lstm_global_${sm}_${rc}_ws${ws}_h${hs}_ns${ns}_ne${ne}_lr${lr}_bs${bs}_nnm${nnmethod}_o${opt}_h${h1}_${h2}_k${k0}_${k1}_${k2}_ne${ne2}_bs${bs2}_lr${lr2}_m${mom}_sub${sub}_pen${pen}_act${act}
										    file=$base.csv
										    qsub -l gpus=1 -o $base.out runLSTM_1b.sh $path -1 $ep $sm $rc lstm $hs $ns $ne $bs $lr $ws $nnmethod $opt $h1 $h2 $k0 $k1 $k2 $ne2 $bs2 $lr2 $mom $sub $pen $act
										done
									    done
									done
								    done
								done
							    done
							done
						    done
						done
					    done
					done
				    done
				done
			    done
			done
		    done
		done
	    done
	done
    done
done
