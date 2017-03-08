#expIdx: the experiment id; performance will be saved under /trainedmodel/webqsp$expIdx$Record

for batchSize in $(seq 20 30 100)
do
	for dropoutP in $(seq 0 0.1 0.4)
	do
		CUDA_VISIBLE_DEVICES=4 th main.lua -task webqsp -model compAggWebqspQs -mem_dim 300 -comp_type submul -batch_size $batchSize -max_epochs 20 -dropoutP $dropoutP -gpu 1 -learning_rate 0.002 -expIdx 0
	done
done


for batchSize in $(seq 20 30 100)
do
	for dropoutP in $(seq 0 0.1 0.4)
	do
		CUDA_VISIBLE_DEVICES=4 th main.lua -task webqsp -model compAggWebqspQsPos -mem_dim 300 -comp_type submul -batch_size $batchSize -max_epochs 20 -dropoutP $dropoutP -gpu 1 -learning_rate 0.002 -expIdx 0
	done
done
