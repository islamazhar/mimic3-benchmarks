from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import imp
import re
import copy

from mimic3models.decompensation import utils
from mimic3benchmark.readers import DecompensationReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

from keras.callbacks import ModelCheckpoint, CSVLogger

# deep supervision
# dropuout rate
# Bidirectional_
# BatchGenDeepSupervision
# form the Discretizer header they cont_channels and those continued channels are going to be normalize..
parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--deep_supervision', dest='deep_supervision', action='store_true')
parser.add_argument('--data', type=str, help='Path to the data of decompensation task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/decompensation/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
parser.set_defaults(deep_supervision=False)
args = parser.parse_args()
print(args)

if args.small_part:
    args.save_every = 2**30

# Build readers, discretizers, normalizers
if args.deep_supervision:
    train_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data, 'train'),
                                                               listfile=os.path.join(args.data, 'train_listfile.csv'),
                                                               small_part=args.small_part)
    val_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data, 'train'),
                                                             listfile=os.path.join(args.data, 'val_listfile.csv'),
                                                             small_part=args.small_part)
else:
    train_reader = DecompensationReader(dataset_dir=os.path.join(args.data, 'train'),
                                        listfile=os.path.join(args.data, 'train_listfile.csv'))
    val_reader = DecompensationReader(dataset_dir=os.path.join(args.data, 'train'),
                                      listfile=os.path.join(args.data, 'val_listfile.csv'))

discretizer = Discretizer(timestep=args.timestep,
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')
# Mazhar: discretizer_header is used in args and also to set cont_channels --> normalizer...
# Mazhar: if normalizer is None then use the previosuly save file?
if args.deep_supervision:
    discretizer_header = discretizer.transform(train_data_loader._data["X"][0])[1].split(',')
else:
    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'decomp_ts{}.input_str:previous.n1e5.start_time:zero.normalizer'.format(args.timestep)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state) # where the normalizer is used?

args_dict = dict(args._get_kwargs()) # Mazhar: getting all the command line arguments...
args_dict['header'] = discretizer_header # Mazhar:  `discretizer_header` is taken from train_data_loader (if deep supervision is on) otherwise from train_reader.
args_dict['task'] = 'decomp'


# Build the model
print("==> using model {}".format(args.network))
model_module = imp.load_source(os.path.basename(args.network), args.network) # loading the args.network module dynamically...what is the value of args.network?
model = model_module.Network(**args_dict)
# dim, batch_norm, dropout, rec_dropout, task, target_repl=False, deep_supervision=False, num_classes=1, depth=1, input_dim=76, **kwargs
suffix = "{}.bs{}{}{}.ts{}".format("" if not args.deep_supervision else ".dsup",
                                   args.batch_size,
                                   ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                   ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                   args.timestep)
model.final_name = args.prefix + model.say_name() + suffix
print("==> model.final_name:", model.final_name)
#> model.final_name: k_lstm.n128.dep1.bs8.ts1.0

# Compile the model
print("==> compiling the model")
optimizer_config = {'class_name': args.optimizer,
                    'config': {'lr': args.lr,
                               'beta_1': args.beta_1}}
# Adam is an adaptive learning rate (optimizer...)

# NOTE: one can use binary_crossentropy even for (B, T, C) shape.
#       It will calculate binary_crossentropies for each class
#       and then take the mean over axis=-1. Tre results is (B, T).
model.compile(optimizer=optimizer_config,
              loss='binary_crossentropy') # standard lstm code
model.summary()
#namespace(batch_norm=False, batch_size=8, beta_1=0.9, data='/home/danfeng/project/mimic3-benchmarks/mimic3models/decompensation/../../data/decompensation/', deep_supervision=False, depth=1, dim=128, dropout=0.0, epochs=100, imputation='previous', l1=0, l2=0, load_state='', lr=0.001, mode='train', network='mimic3models/keras_models/lstm.py', normalizer_state=None, optimizer='adam', output_dir='mimic3models/decompensation', prefix='', rec_dropout=0.0, save_every=1, size_coef=4.0, small_part=False, timestep=1.0, verbose=2)
# Load model weights
n_trained_chunks = 0 # what?
if args.load_state != "": # what is load_state?
    model.load_weights(args.load_state)
    n_trained_chunks = int(re.match(".*chunk([0-9]+).*", args.load_state).group(1))

# Load data and prepare generators
if args.deep_supervision:
    train_data_gen = utils.BatchGenDeepSupervision(train_data_loader, discretizer,
                                                   normalizer, args.batch_size, shuffle=True)
    val_data_gen = utils.BatchGenDeepSupervision(val_data_loader, discretizer,
                                                 normalizer, args.batch_size, shuffle=False)
else:
    # Set number of batches in one epoch
    train_nbatches = 2000
    val_nbatches = 1000 # Why 1000
    if args.small_part:
        train_nbatches = 40
        val_nbatches = 40
    train_data_gen = utils.BatchGen(train_reader, discretizer,
                                    normalizer, args.batch_size, train_nbatches, True, 1, 1)
    val_data_gen = utils.BatchGen(val_reader, discretizer,
                                  normalizer, args.batch_size, val_nbatches, False)
# print(type(train_data_gen))

class_1 = 0
class_0 = 0
for i in range(train_data_gen.steps):
    ret = next(train_data_gen)
    for c in ret[1]:
        if c == 0: class_0 += 1
        else: class_1 += 1
print("Number of class 1 instances= ", class_1, "Number of class 0 instances= ", class_0)
# calculating frequency
# class 0 =  6.382435537401073e-05 | class 1  =  0.0029940119760479044
train_reader = DecompensationReader(dataset_dir=os.path.join(args.data, 'train'),listfile=os.path.join(args.data, 'train_listfile.csv'))
train_data_gen = utils.BatchGen(train_reader, discretizer, normalizer, args.batch_size, train_nbatches, True, 1/class_1, 1/class_0)

if args.mode == 'train':

    # Prepare training
    path = os.path.join(args.output_dir, 'keras_states/' + model.final_name + '.chunk{epoch}.test{val_loss}.state')

    metrics_callback = keras_utils.DecompensationMetrics(train_data_gen=train_data_gen,
                                                         val_data_gen=val_data_gen,
                                                         deep_supervision=args.deep_supervision,
                                                         batch_size=args.batch_size,
                                                         verbose=args.verbose)
    # make sure save directory exists
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    saver = ModelCheckpoint(path, verbose=1, period=args.save_every)

    keras_logs = os.path.join(args.output_dir, 'keras_logs')
    if not os.path.exists(keras_logs):
        os.makedirs(keras_logs)
    csv_logger = CSVLogger(os.path.join(keras_logs, model.final_name + '.csv'),
                           append=True, separator=';')

    print("==> training")
    model.fit_generator(generator=train_data_gen,
                        steps_per_epoch=train_data_gen.steps,
                        validation_data=val_data_gen,
                        validation_steps=val_data_gen.steps,
                        epochs=n_trained_chunks + args.epochs,
                        initial_epoch=n_trained_chunks,
                        callbacks=[metrics_callback, saver, csv_logger],
                        verbose=args.verbose) # the actual params... # https://www.geeksforgeeks.org/keras-fit-and-keras-fit_generator/

elif args.mode == 'test':

    # ensure that the code uses test_reader
    del train_data_gen
    del val_data_gen

    names = []
    ts = []
    labels = []
    predictions = []

    if args.deep_supervision:
        del train_data_loader
        del val_data_loader
        test_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data, 'test'),
                                                                  listfile=os.path.join(args.data, 'test_listfile.csv'),
                                                                  small_part=args.small_part)
        test_data_gen = utils.BatchGenDeepSupervision(test_data_loader, discretizer,
                                                      normalizer, args.batch_size,
                                                      shuffle=False, return_names=True)

        for i in range(test_data_gen.steps):
            print("\tdone {}/{}".format(i, test_data_gen.steps), end='\r') # Done? should it be predicting?
            ret = next(test_data_gen)
            (x, y) = ret["data"]
            cur_names = np.array(ret["names"]).repeat(x[0].shape[1], axis=-1)
            cur_ts = ret["ts"]
            for single_ts in cur_ts:
                ts += single_ts

            pred = model.predict(x, batch_size=args.batch_size)
            for m, t, p, name in zip(x[1].flatten(), y.flatten(), pred.flatten(), cur_names.flatten()):
                if np.equal(m, 1):
                    labels.append(t)
                    predictions.append(p)
                    names.append(name)
        print('\n')
    else: # what if small_part is On now??
        del train_reader
        del val_reader
        test_reader = DecompensationReader(dataset_dir=os.path.join(args.data, 'test'),
                                           listfile=os.path.join(args.data, 'test_listfile.csv'))
        if args.small_part:
            test_nbatches = 40
            #val_nbatches = 40

        test_data_gen = utils.BatchGen(test_reader, discretizer,
                                       normalizer, args.batch_size, test_nbatches
                                       ,shuffle=False, return_names=True)  # put steps = None for a full test

        for i in range(test_data_gen.steps):
            print("predicting {} / {}".format(i, test_data_gen.steps), end='\r')
            ret = next(test_data_gen)
            x, y = ret["data"]
            cur_names = ret["names"]
            cur_ts = ret["ts"]

            x = np.array(x)
            pred = model.predict_on_batch(x)[:, 0] # when deep supervision is ON it is model.predict(x, batch_size=args.batch_size)
            predictions += list(pred)
            labels += list(y)
            names += list(cur_names)
            ts += list(cur_ts)

    metrics.print_metrics_binary(labels, predictions)
    path = os.path.join(args.output_dir, 'test_predictions', os.path.basename(args.load_state)) + '.csv'
    utils.save_results(names, ts, predictions, labels, path)

else:
    raise ValueError("Wrong value for args.mode")
