# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx

from common.data import HybridTrainPipe, HybridValPipe
from nvidia.dali.plugin.mxnet import DALIClassificationIterator


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train imagenet-1k",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    # use a large aug level
    data.set_data_aug_level(parser, 3)
    parser.set_defaults(
        # network
        network          = 'resnet',
        num_layers       = 50,
        # data
        num_classes      = 1000,
        num_examples     = 1281167,
        image_shape      = '3,224,224',
        min_random_scale = 1, # if input image has min size k, suggest to use
                              # 256.0/x, e.g. 0.533 for 480
        # train
        num_epochs       = 80,
        lr_step_epochs   = '30,60',
        dtype            = 'float32'
    )
    parser.add_argument('--use-dali', action='store_true', default=False, \
                         help='Use DALI for input pipeline')
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))
    # train
    print(len(args.gpus))
    if args.use_dali == False: 
        fit.fit(args, sym, data.get_rec_iter)
    else:
        N = len(args.gpus.split(','))
        if args.data_train.endswith(".rec"):
            args.data_train = os.path.dirname(args.data_train)
        if args.data_val.endswith(".rec"):
            args.data_val = os.path.dirname(args.data_val)
        trainpipes = [HybridTrainPipe(batch_size=args.batch_size//N, db_folder = args.data_train, num_threads=2, device_id = i, num_gpus = N) for i in range(N)]
        valpipes = [HybridValPipe(batch_size=args.batch_size//N, db_folder = args.data_val, num_threads=2, device_id = i, num_gpus = N) for i in range(N)]
        
        trainpipes[0].build()
        valpipes[0].build()
        
        dali_train_iter = DALIClassificationIterator(trainpipes, trainpipes[0].epoch_size("Reader"))
        dali_val_iter = DALIClassificationIterator(valpipes, valpipes[0].epoch_size("Reader"))
        #dali_val_iter = DALIClassificationIterator(trainpipes, trainpipes[0].epoch_size("Reader"))
        def get_dali_iter(args, kv=None):
            return (dali_train_iter, dali_val_iter) 
    
        fit.fit(args, sym, get_dali_iter)
