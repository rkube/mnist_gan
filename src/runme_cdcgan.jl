using mnist_gan: get_cdcgan_discriminator, get_cdcgan_generator, train_dscr!, train_gen_cdcgan!, gan_image_output
using MLDatasets: MNIST
using Flux.Data: DataLoader
using Flux
using CUDA
using Zygote
using Logging
using TensorBoardLogger
using ArgParse
using Plots
using Printf

"""
    run_mnist_gan

Trains a conditional deep-convolutional GAN on the MNIST dataset

This is adapted from the Keras example here:
https://keras.io/examples/generative/conditional_gan

Generative Adversarial Networks by I. Goodfellow et al. 2014 https://arxiv.org/abs/1406.2661

"""

CUDA.allowscalar(false)

s = ArgParseSettings()
@add_arg_table s begin
    "--lr_dscr"
        help = "Learning rate for the discriminator. Default = 0.0002"
        arg_type = Float64
        default = 0.0002
    "--lr_gen"
        help = "Learning rate for the generator. Default = 0.0002"
        arg_type = Float64
        default = 0.0002
    "--batch_size"
        help = "Batch size. Default = 128"
        arg_type = Int
        default = 128
    "--latent_dim"
        help = "Size of the latent dimension. Default = 100"
        arg_type = Int
        default = 100
    "--optimizer"
        help = "Optimizer for both, generator and discriminator. Defaults to ADAM"
        arg_type = String
        default = "ADAM"
    "--activation"
        help = "Activation function. Defaults to leakyrelu with 0.2"
        arg_type = String
        default = "relu"
   "--activation_alpha"
        help = "Optional parameter for activation function, α in leakyrelu, celu, elu, etc."
        arg_type = Float64
        default = 0.1
   "--num_epochs"
        help = "Number of epochs to train over"
        arg_type = Int
        default = 10
   "--output_period"
        help = "Period between graphical output of the generator, in units of training iterations."
        arg_type = Int
        default = 100
end

args = parse_args(s)


for (arg, val) in args
    println("   $arg => $val")
end

# Instantiate TensorBoardLogger
#tb_logger = TBLogger("logs/testlog")
#with_logger(tb_logger) do
#    @info "hyperparameters" args
#end
# Number of features per MNIST sample
n_features = 28*28

# Load MNIST train and test data
train_x, train_y = MNIST.traindata(Float32);
test_x, test_y = MNIST.testdata(Float32);

# This dataset has pixel values ∈ [0:1]. Map these to [-1:1]
# See GAN hacks: https://github.com/soumith/ganhacks
train_x = 2f0 * reshape(train_x, 28, 28, 1, :) .- 1f0 |>gpu;
test_x = 2f0 * reshape(test_x, 28, 28, 1, :) .- 1f0 |> gpu;

train_y = Flux.onehotbatch(train_y, 0:9);
test_y = Flux.onehotbatch(test_y, 0:9);

# Insert the train images and labels into a DataLoader
train_loader = DataLoader((data=train_x, label=train_y), batchsize=args["batch_size"], shuffle=true);

# Define the discriminator network.
# The networks takes a flattened 28x28=784 image as input and outputs the
# probability of the image belonging to the real dataset.
discriminator = get_cdcgan_discriminator_v2(args) |> gpu;

# The generator will generate images which come from the learned
# distribution. The output layer has a tanh activation function
# which maps the output to [-1:1], the same range as in the
# pre-processed MNIST images
generator = get_cdcgan_generator_v2(args) |> gpu;

# Optimizer for the discriminator
opt_dscr = getfield(Flux, Symbol(args["optimizer"]))(args["lr_dscr"]);
opt_gen = getfield(Flux, Symbol(args["optimizer"]))(args["lr_gen"]);
# Extract the parameters of the discriminator and generator
ps_dscr = Flux.params(discriminator);
ps_gen = Flux.params(generator);

println("Entering training loop")
lossvec_gen = zeros(args["num_epochs"]);
lossvec_dscr = zeros(args["num_epochs"]);

# This loop follows the algorithm described in Goodfellow et al. 2014
# for number of training iterations
#with_logger(tb_logger) do
    for n in 1:args["num_epochs"]
        loss_sum_gen = 0.0f0
        loss_sum_dscr = 0.0f0

        for (x, y) in train_loader
            # Samples in the current batch, handles edge case
            this_batch = size(x)[end]

            # Since we are employing a purely convolutional architecture, the one-hot
            # labels need to be re-shaped into the same dimension as the image.
            # That is, each label is now a 28x28x10 tensor where the 28x28 image
            # of the hot channel is one, the images in the other channels are zero.
            # 
            # Since repeating does not work for CuArrays we are still working with Matrices here
            labels = reshape(repeat(y, inner=(28*28,1)), (28, 28, 10, this_batch)) |> gpu;
            # Concatenate the labels with the real images along the channel dimension
            real_data = cat(x, labels, dims=3);

            # Sample noise, concatenate with target labels and feed this to the generator
            noise = randn(args["latent_dim"], this_batch) |> gpu;
            random_vector_labels = cat(noise, y, dims=1);
            fake_data = generator(random_vector_labels);
            
            # To pass the real_data and fake_data to the generator we still need to attach the labels in tensor shape to the generated data.
            fake_data = cat(fake_data, labels, dims=3);

            # Update the discriminator by ascending its stochastic gradient
            loss_dscr = train_dscr!(discriminator, real_data, fake_data, ps_dscr, opt_dscr);
            loss_sum_dscr += loss_dscr

            loss_gen = train_gen_cdcgan!(discriminator, generator, args["latent_dim"], y, ps_gen, opt_gen, this_batch)
            loss_sum_gen += loss_gen
        end
        # Add the per-sample loss of the generator and discriminator
        # The train_...! function call binarycrossentropy where the default is agg=mean. That is,
        # these function already return the per-sample loss. In the loop above we aggregate only
        # over the number of batches. So we need to divide by the number of batches, given by
        # length(train_loader)
        lossvec_gen[n] = loss_sum_gen / length(train_loader)
        lossvec_dscr[n] = loss_sum_dscr / length(train_loader)
        @info "end of epoch" n loss_generator=lossvec_gen[n] loss_discriminator=lossvec_dscr[n]

        if(n % args["output_period"] == 0)
            image_name = @sprintf "cdcgan_images_%02d.png" n
            gan_image_output(generator, image_name, args)
        end
    end # epoch
#end # Logger
