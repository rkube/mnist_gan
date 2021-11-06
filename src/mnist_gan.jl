module mnist_gan

using Flux
using Zygote

# All GAN models
include("models.jl")
# Functions used for training
include("training.jl")
end #module
