import os


cmdline = "python train_imagenet.py --gpu %s --batch-size %d --num-epochs 1 --data-train /data/imagenet/train-480-val-256-recordio/train.rec --data-val /data/imagenet/train-480-val-256-recordio/val.rec --disp-batches 100 --network resnet-v1 --num-layers 50 --data-nthreads 32 --min-random-scale 0.533 --max-random-shear-ratio 0 --max-random-rotate-angle 0 --max-random-h 0 --max-random-s 0 --max-random-l 0 "


fp16 = " --dtype float16 "

for fp in ["fp16", "fp32"]:
    for gpu in range(1,5):
        gpus = "".join(str(list(range(gpu)))).replace("[", "").replace("]", "").replace(" ", "")
        cmd = cmdline%(gpus,128*gpu)
        if fp == "fp16":
            cmd += fp16

        os.system(cmd + "--use-dali  2>&1 | tee %sgpu-%s-b256-dali-enable.log"%(gpu,fp))
        os.system(cmd + "2>&1 | tee %sgpu-%s-b256-dali-disable.log"%(gpu,fp))
        print("%sgpu-%s-b256"%(gpu,fp))

print("pass")
