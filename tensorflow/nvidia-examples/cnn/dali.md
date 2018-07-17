# Using TFRecord

|Start Date|End Date  |
|----------|----------|
|2018-11-18|2018-11-19|

## Description

The goal here is to train RN50 using Dali with TFRecord for training only
(i.e. don't use Dali for evaluation).

## Delivrables

- [X] CNN scripts for Dali with TFRecord
- [X] Publish accuracy numbers 
- [X] Publish performance numbers


## Interpretation
Augmentation consists of Resize, Crop, Mirror, and  Normalize.

Now it's training after making labels to be 0-based index.

### Resize, Crop, and Mirror
TF uses `sample_distorted_bounding_box` to get a random bbox with a random size
which is then sliced out of the image using `tf.slice`. Then TF uses
`tf.image.resize_images` to set the size of the image to the right size.
`sample_distorted_bounding_box` supports many features.

Currently with Dali, we first resize (using `Resize`) the image to
some large sizes (256x480) and then crop it with the exact right size
using `CropMirrorNormalize`. This also does the mirroring.
In order to keep the augmentation similar to TF,
we should instead use `RandomResizedCrop` and then use `cropMirrorNormalize`
to resize and mirror the image. Not sure if this will be exactly similar to TF.

### Normalize
TF does this in `runner.py` instead of `image_processing.py` which is the right
place for augmentation. With Dali, we should make sure we skip the TF normalize.

### Shuffle and seed
We should use shuffle and seed the same way that TF uses.

### Mean and STD
TF uses `[121, 115, 100]` for mean and `[70, 68, 71]` for STD.
We should use the same with Dali instead of
mean `[128., 128., 128.]` and STD `[1., 1., 1.]`.

### Evaluation
Currently evaluation with Dali hangs (or perhaps it keeps generating data).
Also `_cnn_model_function` in `runner.py` currently does something
wrong for evaluation when `use_dali=True`.

### Loss
We were getting NaN or 0 loss at the beginning of training.
We fixed a couple of bugs to fix the problem with loss:
1) Making labels 0-based indexed (TFRecord stores 1-based index).
2) Do not normalize images twice.

Currently loss doesn't decrease as fast as the non-Dali case.

## Conclusion

Top-1 accuracy: 66.6700005531 %

Top-5 accuracy: 87.4019980431 %

Loss at epoch 90 1.133  1.776

Performance: 5630

logdir: /mnt/shared/pdavoodi/dali/dali-01



# Improve accuracy of RN50

|Start Date|End Date  |
|----------|----------|
|2018-11-20|          |

## Description

Achieve 75% top1 accuracy for RN50 with Dali.

## Delivrables

- [] CNN scripts for Dali with good accuracy

## Experiments

|Experiment                               |Top-1     |Top-5     |
|-----------------------------------------|----------|----------|
|TF without dali                          |75.3      |92.6      |
|TF without dali, shard images not files  |75.6      |92.8      |
|Sharding dataset                         |71.1      |90.1      |
|Added mirror                             |71.3      |90.3      |
|resize-crop + shfl 10k                   |73.8      |91.6      |
|shuffle files (no seed)                  |75.0      |92.4      |
|shuffle files (seeded)                   |73.7      |91.6      |
|index fixed with shuffle files (no seed) |74.9      |92.4      |
|index fixed without shuffle files        |73.8      |91.6      |
|Reordered files to emulate TF sharding   |73.4      |91.4      |


## Interpretation
Current low accuracy is probably due to different augmentation.
We should make the augmentation similar to TF.

After switching to RandomResizedCrop and increasing shuffle buffer size,
we have reached 73.8%.

After shuffling files, the accuracy increased to 75.05%, but that's only if we
don't set the numpy random seed. 
I guess this imporve in accuracy is due to the following statistical thing:
If different processes look at an image multiple times during an epoch and
drop some images during an epoch, the accuracy will be better
if we train for many epochs.

After removing shuffling in TF and dali, and make TF deterministic,
then GPU0 has the same images in both TF and dali, but GPU1 has
different images between TF and dali.
This could be because of the difference between how TF and dali shard the data.

Found that the index files of 4 TFRecord files were wrong
(2 had no indexes, and 2 had too few). After fixing the indexes,
the accuracy didn't improve. 

In TF, sharding images is 5% slower than sharding files.
This is expected I think because with sharding images, every worker has to read
all the images in the global mini-batch and then keep its relevant shard.
That means most of the read is redundant.

### Reading
Dali uses TFRecordReader.
In multiple experiments, I have saved the images that come out of the whole
dali pipeline and they seemed good.

The only problem is that the firrst two iterations give me the same pictures
in the whole minibatch. I made a standalone test outside of the TF scripts
using dali with TFRecord and it worked fine. I don't know yet which part of
the TF script is causing this problem.

### Shard
Dali uses TFRecordReader. 

TF shards sequentially and giving each piece to a different shard
as it moves a long the dataset.
Dali shards by splitting the whole dataset into big equal contiguous chunks,
where each chunk becomes a shard.

We don't know yet if this difference can be effective.

The accuracy of TF got a little better after sharding images instead of files.

### Shuffle
Dali uses TFRecordReader.

TF shuffles files at every epoch. I did disable this shuffle and saw no change
in accuracy. So I would say including this in dali wouldn't change the accuracy.

According to Przemek, shuffling of  TF and dali work the same way; it
creates a buffer and reads data sequentially from an input into this buffer.
Then it reads from the random location of this buffer.

I am using 10k for the shuffle buffer size.


### nvJPEGDecoder
By reading the TF and nvJPEG documentation,
I can't find a difference that could hurt accuracy. 
But I don't know about the details of the decompression algorithms. 
TF uses an algorithms called INTEGER_FAST.

### Resize
There are some differences between Resize methods of TF, OpenCV, and Dali.
For small images, the difference is bigger. 
When I look at the outputs, I see slight shifts in the object.

This is the experiment I did:

I have the numpy vector corresponding to every decoded and resized image
(without crop, normalize, and mirror).
I calculated the distance between the numpy vectors of TF and dali images.
Then I calculated the norm of the distance vector and looked at the cases
where the norm is larger than 30. The original size of these images is
either too small or it's rectangular. I did similar experiment with OpenCV
and got similar results.

Methods that I used:
1. `tf.image.decode_jpeg` and `tf.image.resize_images`
2. `dali.ops.nvJPEGDecoder` and `dali.ops.Resize`
3. `cv2.imdecode` and `cv2.resize`

I am not sure yet if these slight differences in the output of different
resize methods is expected.

### Crop
I am using `RandomResizedCrop` for dali
and `sample_distorted_bounding_box` for TF.

TF supports `bounding_boxes` and `min_object_covered` but we are not using them.

TF samples a crop differently from dali. TF first samples an aspect ratio
based on which samples a height. But dali samples an aspect ratio and an area
independently. Thus, dali should have a lower chance of getting a crop in
a certain number of attempts. But more importantly the probability distribution
of the dali sampling is different.

Code for sampling:

TF: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/sample_distorted_bounding_box_op.cc#L110

Dali: https://github.com/NVIDIA/DALI/blob/master/dali/pipeline/operators/resize/random_resized_crop.h#L58

### Normalize
I don't see any difference here.

I use `dali.ops.CropMirrorNormalize` for dali, and use
basic subtraction and multiplication ops for TF.

### Seed
Dali isn't deterministic now due to a bug. So using the seed is not effective.

### dtype
TF I/O pipeline (in order of execution):

|operator                        |output dtype|
|--------------------------------|------------|
|decode                          |uint8       |
|crop (slice)                    |uint8       |
|resize                          |float32     |
|cast                            |uint8       |
|mirror                          |uint8       |
|iterator                        |uint8       |
|cast                            |float16     |
|normalize                       |float16     |

Dali I/O pipeline (in order of execution):

|operator                        |output dtype|
|--------------------------------|------------|
|reader                          |            |
|decode                          |uint8       |
|crop & resize                   |uint8       |
|normalize & mirror              |float32     |
|daliop (TF)                     |float32     |
|cast                            |float16     |


## Conclusion

