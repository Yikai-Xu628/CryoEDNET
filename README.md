# CryoEDNet

## Entry points

| File / function | Purpose |
| --- | --- |
| `distill.py / main()` | Distill VGG19 features using clean training images and save the lightweight feature extractor. |
| `train.py / main()` | Load paired data, configure the network, losses and optimizer, and train the denoiser. |
| `train_epoch()` | Run one PyTorch training epoch, updating model parameters and the learning rate after each batch. |
| `run_deepinv()` | Train the network using the DeepInv Trainer. |
| `BatchScheduledTrainer.compute_loss()` | Use DeepInv to update model parameters, then step the batch-level learning-rate scheduler. |
| `save_checkpoint()` | Save model weights, training statistics, configuration, optimizer and scheduler states. |
| `infer.py / main()` | Denoise NPY/MRCS particle stacks using a checkpoint and restore the output intensity scale. |
| `test_real.py / denoise()` | Apply per-image intensity scaling, run the denoiser and restore each image's original intensity scale. |
| `test_real.py / main()` | Denoise a real NPY/MRCS particle stack. |
| `evaluate.py / main()` | Calculate per-image MSE, PSNR and SSIM against paired clean references. |
| `run_experiment.py / main()` | Run data checks, distillation, denoiser training, inference and evaluation in sequence. |
| `now()` | Return the current time with its time zone. |
| `record()` | Update the experiment stage and execution status. |
| `run()` | Execute one experiment stage and write its output to a log. |
| `assess()` | Compare noisy and denoised images, summarize metrics and generate fixed-example comparison figures. |

## Network modules: cryoednet/model.py

Each module's `__init__()` constructs its layers, and `forward()` performs the forward computation.

| Method | Purpose |
| --- | --- |
| `MultiScale.forward()` | Extract features through three scale branches, concatenate them and fuse the result. |
| `EdgeEnhancement.forward()` | Combine a Laplacian response with learned weighted features through a residual connection. |
| `DetailPreservation.forward()` | Extract detail features, apply adaptive weights and add them to the input features. |
| `CryoEDNet.forward()` | Process the feature modules and subtract the predicted residual from the noisy input. |

## Perceptual features and distillation: cryoednet/perceptual.py

| Class / method | Purpose |
| --- | --- |
| `CBAM.__init__() / forward()` | Construct and apply channel and spatial attention. |
| `FeatureExtractor.__init__()` | Construct three convolutional blocks, CBAM and a residual connection. |
| `FeatureExtractor.stages()` | Return three stages of student features for distillation. |
| `FeatureExtractor.forward()` | Return the final features used by the perceptual loss. |
| `VGGTeacher.__init__()` | Load ImageNet-pretrained VGG19 and freeze its parameters. |
| `VGGTeacher.train()` | Keep the teacher in evaluation mode. |
| `VGGTeacher.forward()` | Repeat grayscale input across three channels, normalize using ImageNet statistics and return selected features. |
| `Distiller.__init__()` | Connect the student, teacher and channel-matching convolutional layers. |
| `Distiller.forward()` | Match feature channels and spatial sizes, then calculate the mean L1 loss over three stages. |
| `load_extractor()` | Validate and load a distilled extractor checkpoint, then freeze the extractor. |

## Losses: cryoednet/losses.py

| Method | Purpose |
| --- | --- |
| `SSIMLoss.__init__()` | Create the Gaussian window used to calculate local image statistics. |
| `SSIMLoss.forward()` | Calculate the structural similarity loss, 1 - SSIM. |
| `CombinedLoss.__init__()` | Configure the frozen feature extractor and SSIM loss. |
| `CombinedLoss.train()` | Change the loss module's mode while keeping the extractor in evaluation mode. |
| `CombinedLoss.components()` | Calculate L1 reconstruction, perceptual, SSIM and frequency-amplitude losses separately. |
| `CombinedLoss.forward()` | Combine these losses with weights 1.0, 0.1, 0.5 and 0.1, and check numerical validity. |
| `CombinedLoss.adapt_model()` | Support the DeepInv loss interface without modifying the model. |

## Data handling: cryoednet/data.py

| Method / function | Purpose |
| --- | --- |
| `Stack.__init__()` | Memory-map NPY/MRC/MRCS inputs and validate their types and dimensions. |
| `Stack.__len__()` | Return the number of particles. |
| `Stack.__getitem__()` | Read selected images, convert them to float32 and check for non-finite values. |
| `Stack.close()` | Close the file and memory mapping. |
| `stack_stats()` | Calculate whole-stack mean, standard deviation and intensity range in batches. |
| `PairDataset.__init__()` | Load clean and noisy stacks and check that their shapes match. |
| `PairDataset.__len__()` | Return the number of paired samples. |
| `PairDataset.__getitem__()` | Standardize paired images using training statistics and optionally apply paired flips or rotations. |
| `normalize()` | Apply the explicitly selected training-statistics, stack-statistics or legacy-remapping intensity transform. |
| `denormalize()` | Apply the corresponding inverse transform to restore the output intensity scale. |

## Utilities: cryoednet/utils.py

| Function | Purpose |
| --- | --- |
| `seed_everything()` | Set random seeds for Python, NumPy and PyTorch. |
| `environment()` | Record Python, CUDA and dependency versions. |
| `sha256()` | Calculate a file checksum. |
| `write_json()` | Write a configuration or result to JSON. |
| `new_output_dir()` | Create an output directory and refuse to overwrite a nonempty directory. |

## Software tests: tests/test_release.py

`FakeTeacher.forward()` provides mock teacher features for software tests only.
`ReleaseTests.setUp()` and `tearDown()` create and clean up temporary test data.
`stats()` calculates test statistics; `fixture_extractor()` provides a test-only extractor;
`train_smoke()` runs a small training check.
The `test_*()` methods check network dimensions, residual computation, ablations,
normalization, gradients, checkpoint loading, training entry points, image I/O and overwrite protection.
