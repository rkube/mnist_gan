using mnist_gan: get_vanilla_discriminator, get_vanilla_generator, train_dscr!, train_gen!
using MLDatasets: MNIST
using Flux.Data: DataLoader
using Flux
#using CUDA
using Zygote
using UnicodePlots
using ArgParse
#using Atom; using Juno; Juno.connect(52578)
"""
    run_mnist_gan

Trains a vanilla GAN on the MNIST dataset

This is adapated from the pytorch tutorial presented here
    https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f

Generative Adversarial Networks by I. Goodfellow et al. 2014 https://arxiv.org/abs/1406.2661

# Arguments

- 'η_d': Learning rate of the discriminator network
- 'η_g': Learning rate of the generator network
- batch_size: Number of images presented to the networks in each batch
- num_epochs: Number of epochs to train the network
- output_period: Period with which to plot generator samples to the terminal
"""



s = ArgParseSettings()
@add_arg_table s begin
    "--lr_dscr"
        help = "Learning rate for the discriminator. Default = 0.0002"
        argtype = Float
        default = 0.0002
    "--lr_gen"
        help = "Learning rate for the generator. Default = 0.0002"
        argtype = Float
        default = 0.0002
    "--batch_size"
        help = "Batch size. Default = 1024"
        argtype = Int
        default = 128
    "--num_epochs"
        help = "Number of epochs to train for. Default = 1000"
        argtype = Int
        default = 1000
    "--latent_dim"
        help = "Size of the latent dimension. Default = 100"
        argtype = Int
        default = 100
    "--optimizer"
        help = "Optimizer for both, generator and discriminator. Defaults to ADAM"
        argtype = String
        default = "ADAM"
    "--activation"
        help = "Activation function. Defaults to leakyrelu with 0.2"
        argtype = String
        default = "leakyrelu"
end

parsed_args = parse_args(s)

# function run_mnist_gan(;η_d=2e-4, η_g=2e-4, batch_size=1024, num_epochs=1000, output_period=100)
# Number of features per MNIST sample
n_features = 28*28
# Latent dimension of the generator

# Load MNIST train and test data
train_x, train_y = MNIST.traindata(Float32);
test_x, test_y = MNIST.testdata(Float32);

# This dataset has pixel values ∈ [0:1]. Map these to [-1:1]
# See GAN hacks: https://github.com/soumith/ganhacks
train_x = 2f0 * reshape(train_x, 28, 28, 1, :) .- 1f0 #|>gpu;
test_x = 2f0 * reshape(test_x, 28, 28, 1, :) .- 1f0 #|> gpu;

train_y = Flux.onehotbatch(train_y, 0:9) #|> gpu;
test_y = Flux.onehotbatch(test_y, 0:9) #|> gpu;

# Insert the train images and labels into a DataLoader
train_loader = DataLoader((data=train_x, label=train_y), batchsize=args["batch_size"], shuffle=true);

# Define the discriminator network.
# The networks takes a flattened 28x28=784 image as input and outputs the
# probability of the image belonging to the real dataset.
# I had a hard time training this when used the default value of a=0.01 in LeakyReLU.I.e.
# the syntax ... Dense(1024, 512, leakyrelu)... did not work well.
# I really need x -> leakyrelu(x, 0.2f0)
#
# discriminator = get_vanilla_discriminator()
#
# # The generator will generate images which come from the learned
# # distribution. The output layer has a tanh activation function
# # which maps the output to [-1:1], the same range as in the
# # pre-processed MNIST images
#
# generator = get_vanilla_generator()
#
# # Optimizer for the discriminator
# opt_dscr = getfield(Flux, Symbol(args["optimizer"]))(args["lr_dscr"])
# opt_gen = getfield(Flux, Symbol(args["optimizer"]))(args["lr_gen"])
#
# println("Entering training loop")
# lossvec_gen = zeros(args["num_epochs"])
# lossvec_dscr = zeros(args["num_epochs"])
#
# for n in 1:num_epochs
#     loss_sum_gen = 0.0f0
#     loss_sum_dscr = 0.0f0
#
#     for (x, y) in train_loader
#         # Samples in the current batch, handles edge case
#         this_batch = size(x)[end]
#         # Train the discriminator
#         # - Flatten the images, which squashes all dimensions and keeps the
#         #   the last dimension, which is the batch dimension
#         real_data = flatten(x);
#
#         # Sample minibatch of m noise examples z¹, …, zᵐ from noise prior pg(z)
#         noise = randn(args["latent_dim"], this_batch) #|> gpu
#         fake_data = generator(noise)
#
#         # Update the discriminator by ascending its stochastic gradient
#         # ∇_theta_d 1\m Σ_{i=1}^{m} [ log D(xⁱ) + log(1 - D(G(zⁱ))]
#         loss_dscr = train_dscr!(discriminator, real_data, fake_data, this_batch)
#         loss_sum_dscr += loss_dscr
#
#         #   Sample minibatch of m noise examples z¹, …, zᵐ from noise prior pg(z)
#         #   Update the generator by descending its stochastic gradient
#         #       ∇_theta_g 1/m Σ_{i=1}^{m} log(1 - D(G(zⁱ))
#         loss_gen = train_gen!(discriminator, generator)
#         loss_sum_gen += loss_gen
#     end
#
#     # Add the per-sample loss of the generator and discriminator
#     lossvec_gen[n] = loss_sum_gen / size(train_x)[end]
#     lossvec_dscr[n] = loss_sum_dscr / size(train_x)[end]
#
#     if n % output_period == 0
#         @show n
#         noise = randn(args["latent_dim"], 4) #|> gpu;
#         fake_data = reshape(generator(noise), 28, 4*28);
#         p = heatmap(fake_data, colormap=:inferno)
#         print(p)
#     end
# end # Training loop]
