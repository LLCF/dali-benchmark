import os


cmdline = "python -m apex.parallel.multiproc perf.py -a resnet50 -b %s --world-size %s --epochs=1 --iterations=100 -j 5 -p 10 --dist-backend nccl ./ "


fp16 = " --fp16 "

for fp in ["fp16", "fp32"]:
    for gpu in range(4,5):
        #gpus = "".join(str(list(range(gpu)))).replace("[", "").replace("]", "").replace(" ", "")
        if fp == "fp16":
            cmd = cmdline%(128, gpu)
            cmd += fp16
        else:
            cmd = cmdline%(64, gpu)

        os.system(cmd + " --use-dali  2>&1 | tee %sgpu-%s-b128-dali-enable.log"%(gpu,fp))
        os.system(cmd + " 2>&1 | tee %sgpu-%s-b128-dali-disable.log"%(gpu,fp))
        print("%sgpu-%s-b256"%(gpu,fp))

print("pass")
