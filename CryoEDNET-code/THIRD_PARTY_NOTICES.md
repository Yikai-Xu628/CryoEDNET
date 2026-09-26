# Sources and dependencies

CryoEDNet was developed by the authors using the DeepInv framework. This release
retains an optional DeepInv training backend and provides a small PyTorch backend
for transparent inspection and tests. The neural modules and losses were
reconstructed from the authors' manuscript and supplied code.

- DeepInv: https://github.com/deepinv/deepinv
  Trainer API: https://deepinv.org/api/stubs/deepinv.Trainer.html
  Original local source identifies version 0.2.1 and license BSD 3-Clause Clear.
  The current upstream LICENSE states BSD 3-Clause. Consult the LICENSE shipped
  with the actual installed distribution. This package does not vendor DeepInv.
- PyTorch: https://github.com/pytorch/pytorch
- Torchvision: https://github.com/pytorch/vision
  VGG19 teacher: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg19.html
- NumPy: https://numpy.org/
- mrcfile: https://github.com/ccpem/mrcfile

Dependency copyrights and licenses remain with their respective owners.
Pretrained VGG19 weights are downloaded by torchvision when distillation is
explicitly run; they are not redistributed in this folder.

Please cite the DeepInv project and its recommended citation when describing the
training framework. Do not describe the underlying framework as authored here.
CryoEDNet has no publication DOI recorded in this package; do not invent one.
