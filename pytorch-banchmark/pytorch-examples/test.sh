#!/bin/bash

AWK=awk
SED=sed
AWK_SCRIPT=./update.awk
PERF_TST=perf.py
NGPUS_AVAIL=$(nvidia-smi -L 2>/dev/null | wc -l)

if [ $NGPUS_AVAIL -eq 0 ] ; then
    echo "No any GPU found"
else
    echo $NGPUS_AVAIL "GPU(s) found:"
    nvidia-smi  -L | $SED -e 's/(.*//'
fi

GPU_NAME=UNKNOWN
if [[ $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | grep "Tesla V100" | wc -l) -gt 0 ]]; then
	GPU_NAME=DGX_V100
    if [ $NGPUS_AVAIL -ge 4 ]; then
        NGPUS_AVAIL=4
    fi
elif [[ $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | grep "Tesla P100" | wc -l) -gt 0 ]]; then
	GPU_NAME=DGX_P100
    if [ $NGPUS_AVAIL -ge 4 ]; then
        NGPUS_AVAIL=4
    fi
elif [[ $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | grep "Tesla M40" | wc -l) -gt 0 ]]; then
	GPU_NAME=TESLA_M40
    if [ $NGPUS_AVAIL -ge 2 ]; then
        NGPUS_AVAIL=2
    fi
elif [[ $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | grep "Tesla P40" | wc -l) -gt 0 ]]; then
	GPU_NAME=TESLA_P40
    if [ $NGPUS_AVAIL -ge 1 ]; then
        NGPUS_AVAIL=1
    fi
fi

cp /opt/pytorch/apex/examples/imagenet/*.py .

cat <<EOF > $AWK_SCRIPT
  BEGIN{add_arg=0; func=""; timer=0; abt = 0}
  /^def (main|train|validate)/{func=substr(\$2,1,4);}
  /^def (train|validate)/{gsub(/\)/,", iterations)", \$0); timer=0;}
  /import time/{ print "import json" }
  /validate\(/{if(func == "main"){gsub(/\)/,", args.iterations)", \$0);}}
  /train\(/   {if(func == "main"){gsub(/\)/,", args.iterations))", \$0);}}
  /train\(/ {if(func=="main"){gsub(/train\(/, "avgBatchTime = max(avgBatchTime, train(", \$0)}}
  /parser.add_argument/{ if(add_arg == 0){
           print "parser.add_argument('--iterations', default=0, type=int, metavar='N',";
           print "                    help='Number of iterations to run per epoch')";
           print "parser.add_argument('--ref', type=str, default=None,";
           print "                    help='path to json file with performance reference')";
           print "parser.add_argument('--create_reference', type=bool, default=False,";
           print "                    help='if true, will create json file with reference perf values under --ref file')";
           add_arg = 1;
  }}
  /model = models.__dict__\[args.arch\]\(\)/{if(func == "main"){
           print "        model = resnet.__dict__[args.arch]()"
           getline
  }}
  {
      print \$0
  }
  /import torch.utils.data/{print "import resnet as resnet"}
  /end = time.time\(\)/{  if(func == "trai"){
       timer = timer + 1;
       if(timer == 2){
           print "\n        if i == 0:\n            batch_time.reset()";
           print "        if (iterations != 0) and (iterations < i):\n            break;";
       }
  }}
  /model.eval\(\)/{  if(func == "vali"){
       print "\n    if iterations != 0:\n        return top1.avg;\n";
       func="";
  }}
  /return/{if (func == "main" && abt == 0){
            print "\n    avgBatchTime = 0"
            abt = 1
  }}
  /\}, is_best\)/{ if (func=="main"){
    print "    imagesPerSecond = int(args.batch_size * args.world_size / avgBatchTime)";
    print "    print(\"##### {}\".format(imagesPerSecond))";
    print "    if args.ref is not None:";
    print "        data=\"{}\"";
    print "        if os.path.isfile(args.ref):";
    print "            with open(args.ref, 'r') as f:";
    print "                data = f.read()";
    print "";
    print "        ref = json.loads(data)";
    print "";
    print "        if args.create_reference:";
    print "            cdc = str(torch.cuda.device_count())";
    print "            bs = str(args.batch_size)";
    print "            if cdc not in ref.keys():";
    print "                ref[cdc] = {}";
    print "            if bs not in ref[cdc].keys():";
    print "                ref[cdc][bs]={}";
    print "            ref[cdc][bs][args.arch]=imagesPerSecond";
    print "            with open(args.ref, 'w') as rf:";
    print "                json.dump(ref, rf)";
    print "        else:";
    print "            try:";
    print "                total_batch = torch.cuda.device_count()*args.batch_size";
    print "                baseline = ref[str(torch.cuda.device_count())][str(total_batch)][args.arch]";
    print "                if imagesPerSecond < baseline * 0.9:";
    print "                    print(\"REGRESSION: {} current average mbs < 90% of baseline mbs ({} mbs)\".format(imagesPerSecond, baseline))";
    print "            except KeyError as _:";
    print "                pass";
  }}

  /top5=top5/ { if (func=="trai"){
    print "";
    print "    return batch_time.avg";
  }}

EOF

$AWK -f $AWK_SCRIPT < main.py > $PERF_TST
if [ ! -d "val" ]; then
   ln -sf /data/imagenet/val-jpeg/ val
fi
if [ ! -d "train" ]; then
   ln -sf /data/imagenet/train-jpeg/ train
fi

export OMP_NUM_THREADS=1

touch out.$$
echo "" > out.$$

failedtests=""
failedtestsN=0

basefile=$GPU_NAME"_perf.json"
echo $basefile
if [ -f $basefile ]; then
    basefile="--ref=$basefile"
else
    basefile=""
fi


cat << EOF > out.$$

ResNet fp32 [imgs/sec]
nGPUs|  mbs/gpu  |  total mbs | Arch:  50  |  101 |  152
---------------------------------------------------------
EOF

ITER=100
NGPUS=1
while( [ $NGPUS -le $NGPUS_AVAIL ] ); do
	echo ""
    echo "#GPU(s) - "$NGPUS | tee -a log
    export CUDA_VISIBLE_DEVICES=$(echo $NGPUS| $AWK "BEGIN{printf \"0\"}{for(i=1;i<\$0;i++){printf \",\"i;}}")
    echo "GPUs: "$CUDA_VISIBLE_DEVICES | tee -a log

    for mbs in 32 64; do
        printf "%3d |%6d | %6d |" $NGPUS $mbs $(($mbs*$NGPUS)) >> out.$$
        for rn in 50 101 152; do
            echo "################################################################################"
            echo "Printing out nvidia-smi and python processes to check for QA hangs..."
            nvidia-smi
            set +e
            ps -aux | grep python
            set -e
            echo "################################################################################"
            benchcmd="python -m apex.parallel.multiproc $PERF_TST -a resnet$rn -b $mbs $basefile --epochs=1 --iterations=$ITER -j 5 -p 10 --dist-backend nccl ./"
            echo $benchcmd
            $benchcmd 2>&1 | tee benchres

            if [[ $(cat benchres | grep "REGRESSION" | wc -l) -ne 0 ]]; then
                failureCondition=$(cat benchres | grep "REGRESSION")
                failedtests="$failedtests| Failed $GPU_NAME NGPUS=$NGPUS mbs=$mbs resnet$rn with $failureCondition|"
                failedtestsN=$(($failedtestsN+1))
            fi
            ips=$(cat benchres | grep "#####" | sed -e 's/^\#*\ *//' )
            echo $ips | $AWK "{s=(\$1); printf \"%10.0f |\",s;}" >> out.$$
        done
        echo "" >> out.$$
    done

    if [ $NGPUS -eq $NGPUS_AVAIL ]; then break; fi
    NGPUS=$(( $NGPUS*2 ))
    if [ $NGPUS -gt $NGPUS_AVAIL ]; then NGPUS=$NGPUS_AVAIL; fi
done

cat out.$$

if [ $failedtestsN -ne 0 ]; then
    echo "$failedtestsN tests failed:"
    echo $failedtests | sed "s/|/\n/g"
    exit 1
fi
