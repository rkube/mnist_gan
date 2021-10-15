module mnist_gan

using MLDatasets: MNIST
using Flux.Data: DataLoader
using Flux
using CUDA
using Zygote
using UnicodePlots

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
function run_mnist_gan(;η_d=2e-4, η_g=2e-4, batch_size=1024, num_epochs=1000, output_period=100)
    # Number of features per MNIST sample
    n_features = 28*28
    # Latent dimension of the generator
    latent_dim = 100

    # Load MNIST train and test data
    train_x, train_y = MNIST.traindata(Float32);
    test_x, test_y = MNIST.testdata(Float32);

    # This dataset has pixel values ∈ [0:1]. Map these to [-1:1]
    # See GAN hacks: https://github.com/soumith/ganhacks
    train_x = 2f0 * reshape(train_x, 28, 28, 1, :) .- 1f0 |>gpu;
    test_x = 2f0 * reshape(test_x, 28, 28, 1, :) .- 1f0 |> gpu;

    train_y = Flux.onehotbatch(train_y, 0:9) |> gpu; 
    test_y = Flux.onehotbatch(test_y, 0:9) |> gpu;

    # Insert the train images and labels into a DataLoader
    train_loader = DataLoader((data=train_x, label=train_y), batchsize=batch_size, shuffle=true);

    # Define the discriminator network. 
    # The networks takes a flattened 28x28=784 image as input and outputs the
    # probability of the image belonging to the real dataset.
    # I had a hard time training this when used the default value of a=0.01 in LeakyReLU.I.e.
    # the syntax ... Dense(1024, 512, leakyrelu)... did not work well. 
    # I really need x -> leakyrelu(x, 0.2f0)
    discriminator = Chain(Dense(n_features, 1024, x -> leakyrelu(x, 0.2f0)),
                          Dropout(0.3),
                          Dense(1024, 512, x -> leakyrelu(x, 0.2f0)),
                          Dropout(0.3),
                          Dense(512, 256, x -> leakyrelu(x, 0.2f0)),
                          Dropout(0.3),
                          Dense(256, 1, sigmoid)) |> gpu

    # The generator will generate images which come from the learned
    # distribution. The output layer has a tanh activation function
    # which maps the output to [-1:1], the same range as in the 
    # pre-processed MNIST images

    generator = Chain(Dense(latent_dim, 256, x -> leakyrelu(x, 0.2f0)),
                      Dense(256, 512, x -> leakyrelu(x, 0.2f0)),
                      Dense(512, 1024, x -> leakyrelu(x, 0.2f0)),
                      Dense(1024, n_features, tanh)) |> gpu


    # Optimizer for the discriminator
    opt_dscr = ADAM(η_d)
    opt_gen = ADAM(η_g)

    function train_dscr!(discriminator, real_data, fake_data, this_batch)
        # Given real and fake data, update the parameters of the discriminator network in-place
        # Assume that size(real_data) = 784xthis_batch
        # this_batch is the number of samples in the current batch
        # Concatenate real and fake data into one big vector
        all_data = hcat(real_data, fake_data)
        # Target vector for predictions
        all_target = [ones(eltype(real_data), 1, this_batch) zeros(eltype(fake_data), 1, this_batch)] |> gpu;

        ps = Flux.params(discriminator)
        loss, back = Zygote.pullback(ps) do
            preds = discriminator(all_data)
            # The documentation says to use logitbinarycrossentropy, but for this case the plain
            # binarycrossentropy works fine
            loss = Flux.Losses.binarycrossentropy(preds, all_target)
        end
        # To get the gradients we evaluate the pullback with 1.0 as a seed gradient.
        grads = back(1f0)

        # Update the parameters of the discriminator with the gradients we calculated above
        Flux.update!(opt_dscr, Flux.params(discriminator), grads)
        
        return loss 
    end
     

    function train_gen!(discriminator, generator)
        # Updates the parameters of the generator in-place
        # Let the generator create fake data which should out-smart the discriminator
        # The discriminator is fooled if it outputs a 1 for the samples generated
        # by the generator.
        noise = randn(latent_dim, batch_size) |> gpu;

        ps = Flux.params(generator)
        # Evaluate the loss function while calculating the pullback. We get the loss for free
        # by manually calling Zygote.pullback.
        loss, back = Zygote.pullback(ps) do
            preds = discriminator(generator(noise));
            loss = Flux.Losses.binarycrossentropy(preds, 1.) 
        end
        # Evaluate the pullback with a seed-gradient of 1.0 to get the gradients for
        # the parameters of the generator
        grads = back(1.0f0)
        Flux.update!(opt_gen, Flux.params(generator), grads)
        return loss
    end

    println("Entering training loop")
    lossvec_gen = zeros(num_epochs)
    lossvec_dscr = zeros(num_epochs)

    for n in 1:num_epochs
        loss_sum_gen = 0.0f0
        loss_sum_dscr = 0.0f0

        for (x, y) in train_loader
            # Samples in the current batch, handles edge case
            this_batch = size(x)[end]
            # Train the discriminator
            # - Flatten the images, which squashes all dimensions and keeps the 
            #   the last dimension, which is the batch dimension
            real_data = flatten(x);

            # Sample minibatch of m noise examples z¹, …, zᵐ from noise prior pg(z)
            noise = randn(latent_dim, this_batch) |> gpu
            fake_data = generator(noise)

            # Update the discriminator by ascending its stochastic gradient
            # ∇_theta_d 1\m Σ_{i=1}^{m} [ log D(xⁱ) + log(1 - D(G(zⁱ))]
            loss_dscr = train_dscr!(discriminator, real_data, fake_data, this_batch)
            loss_sum_dscr += loss_dscr

            #   Sample minibatch of m noise examples z¹, …, zᵐ from noise prior pg(z)
            #   Update the generator by descending its stochastic gradient
            #       ∇_theta_g 1/m Σ_{i=1}^{m} log(1 - D(G(zⁱ))
            loss_gen = train_gen!(discriminator, generator)
            loss_sum_gen += loss_gen
        end

        # Add the per-sample loss of the generator and discriminator
        lossvec_gen[n] = loss_sum_gen / size(train_x)[end]
        lossvec_dscr[n] = loss_sum_dscr / size(train_x)[end]

        if n % output_period == 0
            @show n
            noise = randn(latent_dim, 4) |> gpu;
            fake_data = reshape(generator(noise), 28, 4*28);
            p = heatmap(fake_data, colormap=:inferno)
            print(p)
        end
    end # Training loop
    return lossvec_gen, lossvec_dscr

end

end


