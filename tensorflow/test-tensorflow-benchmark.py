import os


cmdline = "mpiexec --allow-run-as-root --bind-to socket -np %s python -u  nvidia-examples/cnn/resnet.py --layers=50 --data_dir=/data/imagenet/train-val-tfrecord-480 --data_idx_dir=/data/imagenet/train-val-tfrecord-480-idx/ --precision=%s --num_iter=1000  --iter_unit=batch --display_every=50 --batch=%s "



for fp in ["fp16", "fp32"]:
    for gpu in range(1,5):
        if fp == "fp16":
            batch = 128
        else:
            batch = 64
        
        cmd = cmdline%(gpu,fp, batch)
        os.system(cmd + "2>&1 | tee %sgpu-%s-b%s-dali-disable.log"%(gpu,fp, batch))
        os.system(cmd + "--use_dali  2>&1 | tee %sgpu-%s-b%s-dali-enable.log"%(gpu,fp, batch))
        print("%sgpu-%s-b%s"%(gpu,fp, batch))

print("pass")
