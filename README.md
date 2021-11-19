# Learning the MNIST dataset using GANs with Fluxml
You want a [Generative Adversial Network](https://arxiv.org/abs/1406.2661) to learn the MNIST dataset using Julia's
[FluxML](https://www.fluxml.ai) library? This repository has you covered.

Simply clone the repo and run
```
$ julia --project=. -i src/runme.jl
```

The `runme.jl` script parses the command line and can configure various aspects of the
model and other hyperparameters through the command line. This makes it easy to call
it from a shell-script, as is common when running large hyperparameter scans.

To list the parameters run this command

```
$ julia --project=. src/runme.jl --help
```


