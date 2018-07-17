  BEGIN{add_arg=0; func=""; timer=0; abt = 0}
  /^def (main|train|validate)/{func=substr($2,1,4);}
  /^def (train|validate)/{gsub(/\)/,", iterations)", $0); timer=0;}
  /import time/{ print "import json" }
  /validate\(/{if(func == "main"){gsub(/\)/,", args.iterations)", $0);}}
  /train\(/   {if(func == "main"){gsub(/\)/,", args.iterations))", $0);}}
  /train\(/ {if(func=="main"){gsub(/train\(/, "avgBatchTime = max(avgBatchTime, train(", $0)}}
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
      print $0
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

