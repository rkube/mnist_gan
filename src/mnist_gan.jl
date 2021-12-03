module mnist_gan

using NNlib
using Flux
using Zygote
using Base:Fix2

# All GAN models
include("models.jl")
# Functions used for training
include("training.jl")
# Functions for output
include("output.jl")
end #module
