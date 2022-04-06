# Latent Flow: Neural flow for SDE
---
We model data as the solutions to SDEs using neural flow. This allows more efficient computations than using numerical solvers.

## Installation
---
Install the local package nfsde 
```
pip install -e .
```

## Generate data
---
Generate toy datasets.
```
python -m nfsde.experiments.synthetic.generate
```



## Experiments
---
### Training models
We train Latent Flow, Latent SDE and Latent CTFP on stochastic gompertzian datasets. Trained models are saved in result/{model}/{data}. To train Latent Flow:
```
python -m nfsde.train --model flow-mc --data gompertzian --epochs 1000 --batch-size 100 --weight-decay 1e-5 --flow-model resnet --flow-layers 4 --time-net TimeFourier --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 128 --activation ReLU --final-activation Identity --base-sde combine --hidden-state-dim 16 --w-dim 10 --z-dim 0 --flow-dim 4 --encoder-hidden-state-dim 16 --encoder-hidden-layers 2 --encoder-hidden-dim 64 --iwae-train 25 --iwae-test 50 --early-stop
```

To train Latent SDE:
```
python -m nfsde.train --model latent-sde --data gompertzian --epochs 1000 --batch-size 100 --weight-decay 1e-5 --hidden-layers 2 --hidden-dim 128 --activation Softplus --final-activation Tanh --hidden-state-dim 16 --w-dim 10 --z-dim 20 --encoder-hidden-state-dim 16 --encoder-hidden-layers 2 --encoder-hidden-dim 64 --iwae-train 25 --iwae-test 50 --early-stop
```

To train Latent CTFP with a coupling flow decoder:
```
python -m nfsde.train --model ctfp-flow --data gompertzian --epochs 1000 --batch-size 100 --weight-decay 1e-5 --flow-model coupling --flow-layers 4 --time-net TimeFourier --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 128 --activation ReLU --final-activation Identity --base-sde brownian --hidden-state-dim 16 --encoder-hidden-state-dim 32 --encoder-hidden-layers 2 --encoder-hidden-dim 64 --iwae-train 25 --iwae-test 50 --early-stop
```


### Test metric: Train on synthetic test on real (TSTR)
We measure the performance of the models by training a latent ODE on the generated data from models, and then test it on real data.
```
python -m nfsde.train_synth_test_real --data gbm --generator-path path/to/model --model flow-mc --epochs 1000 --weight-decay 1e-5 --batch-size 100 --hidden-layers 2 --hidden-dim 64 --encoder-hidden-state-dim 32 --encoder-hidden-layers 2 --encoder-hidden-dim 64 --encoder-hidden-layers 2 --z-dim 0
```

### GAN methods
In our experiments, GAN methods do not fit complicated data. 
To run SDE-GAN:
```
python -m nfsde.train --model sde-gan --data ou2 --epochs 1000 --batch-size 1024 --d-hidden-layers 1 --d-hidden-dim 16 --hidden-dim 16 --g-lr 2e-4 --hidden-state-dim 32
```


To run Flow-GAN:
```
python -m nfsde.train --model flow-gan --data ou2 --epochs 1000 --batch-size 1024 --weight-decay 1e-5 --flow-model resnet --flow-layers 1 --time-net TimeFourier --time-hidden-dim 8 --hidden-layers 2 --hidden-dim 64 --d-hidden-layers 1 --d-hidden-dim 16 --activation ReLU --final-activation Identity --hidden-state-dim 16 --w-dim 4 --z-dim 4 --d-model CDE 
```