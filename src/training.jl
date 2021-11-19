
"""
Function for training the discriminator and generator
"""

function train_dscr!(discriminator, real_data, fake_data, ps_dscr, opt_dscr)
    # Given real and fake data, update the parameters of the discriminator network in-place
    # Assume that size(real_data) = 784xthis_batch
    # this_batch is the number of samples in the current batch
    # Concatenate real and fake data into one big vector along the batch dimension
    #all_data = hcat(real_data, fake_data)
    all_data = cat(real_data, fake_data, dims=ndims(real_data))
    this_batch = size(real_data)[end]
    # Target vector for predictions
    all_target = [ones(eltype(real_data), 1, this_batch) zeros(eltype(fake_data), 1, this_batch)] |> gpu;

    loss, back = Zygote.pullback(ps_dscr) do
        preds = discriminator(all_data)
        # The documentation says to use logitbinarycrossentropy, but for this case the plain
        # binarycrossentropy works fine
        loss = Flux.Losses.binarycrossentropy(preds, all_target)
    end
    # To get the gradients we evaluate the pullback with 1.0 as a seed gradient.
    grads = back(1f0)

    # Update the parameters of the discriminator with the gradients we calculated above
    Flux.update!(opt_dscr, ps_dscr, grads)

    return loss
end


function train_gen!(discriminator, generator, latent_dim, ps_gen, opt_gen, batch_size)
    # Updates the parameters of the generator in-place
    # Let the generator create fake data which should out-smart the discriminator
    # The discriminator is fooled if it outputs a 1 for the samples generated
    # by the generator.
    noise = randn(latent_dim, batch_size) |> gpu;

    # Evaluate the loss function while calculating the pullback. We get the loss for free
    # by manually calling Zygote.pullback.
    loss, back = Zygote.pullback(ps_gen) do
        preds = discriminator(generator(noise));
        loss = Flux.Losses.binarycrossentropy(preds, 1.)
    end
    # Evaluate the pullback with a seed-gradient of 1.0 to get the gradients for
    # the parameters of the generator
    grads = back(1.0f0)
    Flux.update!(opt_gen, ps_gen, grads)
    return loss
end


function train_gen_cdcgan!(discriminator, generator, latent_dim, one_hot_labels, ps_gen, opt_gen, batch_size)
    # Updates the parameters of the generator in-place
    # Almost identical to train_gen!, but this function concatenates the batch labels
    # with random noise
    noise = cat(randn(latent_dim, batch_size), one_hot_labels, dims=1) |> gpu;
    labels = reshape(repeat(one_hot_labels, inner=(28*28, 1)), (28, 28, 10, batch_size)) |> gpu;

    # Evaluate the loss function while calculating the pullback. We get the loss for free
    # by manually calling Zygote.pullback.
    loss, back = Zygote.pullback(ps_gen) do
        fake_images = generator(noise);
        # The generator outputs the images but the discriminator also expects the label encoding 
        # as separate channels.
        fake_images_and_labels = cat(fake_images, labels, dims=3);
        preds = discriminator(fake_images_and_labels);
        loss = Flux.Losses.binarycrossentropy(preds, 1.)
    end
    # Evaluate the pullback with a seed-gradient of 1.0 to get the gradients for
    # the parameters of the generator
    grads = back(1.0f0)
    Flux.update!(opt_gen, ps_gen, grads)
    return loss
end
