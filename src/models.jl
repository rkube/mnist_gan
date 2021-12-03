
"""
Contains functions that return a model.
"""

function get_vanilla_discriminator(args)
    # Construct the activation function from the dictionary args and handle the edge-case when the
    # activation function can take an additional parameter α. This is the case f.ex. for leakyrelu, 
    # elu, and celu.
    # The combination of Symbol and getfield allow to map from the string of a function name to the
    # actual function.
    # F.ex. getfield(Main, Symbol("sin")) returns the function `sin`.
    # First, let's make sure that we are working in Float32 here and set both α and act to const.
    # See for example here: https://m3g.github.io/JuliaNotes.jl/stable/closures/
    # And the discussion in slack: https://julialang.slack.com/archives/C689Y34LE/p1636215205114200

    # If the activation function can accept a parameter we create a closure so that the
    # call syntax in the Dense(...,..., act) is the same for these functions and activation
    # function that only take one argument, as relu.
    if args["activation"] in ["celu", "elu", "leakyrelu", "trelu"]
        # Now continue: We want to use Base.Fix2
        act = Fix2(getfield(NNlib, Symbol(args["activation"])), Float32(args["activation_alpha"]))
    else
        act = getfield(NNlib, Symbol(args["activation"]));
    end

    return Chain(Dense(28 * 28, 1024, act),
                 Dropout(args["prob_dropout"]),
                 Dense(1024, 512, act),
                 Dropout(args["prob_dropout"]),
                 Dense(512, 256, act),
                 Dropout(args["prob_dropout"]),
                 Dense(256, 1, sigmoid)) |> gpu
end


function get_vanilla_critic(args)
    # A critic to be used in Wasserstein GANs
    if args["activation"] in ["celu", "elu", "leakyrelu", "trelu"]
        act = Fix2(getfield(NNlib, Symbol(args["activation"])), Float32(args["activation_alpha"]))
    else
        act = getfield(NNlib, Symbol(args["activation"]));
    end

    return Chain(Dense(28 * 28, 1024, act),
                 Dropout(args["prob_dropout"]),
                 Dense(1024, 512, act),
                 Dropout(args["prob_dropout"]),
                 Dense(512, 256, act),
                 Dropout(args["prob_dropout"]),
                 Dense(256, 1)) |> gpu
end


function get_vanilla_generator(args)
    # The logic of this function is the same as in get_vanilla_discriminator
    if args["activation"] in ["celu", "elu", "leakyrelu", "trelu"]
        act = Fix2(getfield(NNlib, Symbol(args["activation"])), Float32(args["activation_alpha"]))
    else
        act = getfield(NNlib, Symbol(args["activation"]));
    end

    return Chain(Dense(args["latent_dim"], 256, act),
                 Dense(256, 512, act),
                 Dense(512, 1024, act),
                 Dense(1024, 28*28, tanh)) |> gpu
end


function get_cdcgan_discriminator(args)
    if args["activation"] in ["celu", "elu", "leakyrelu", "trelu"]
        # Now continue: We want to use Base.Fix2
        act = Fix2(getfield(NNlib, Symbol(args["activation"])), Float32(args["activation_alpha"]))
    else
        act = getfield(NNlib, Symbol(args["activation"]));
    end
    
    return Chain(Conv((3, 3), 11 => 32, act),
                 Conv((5, 5), 32 => 64, act),
                 MaxPool((2, 2)),
                 x -> flatten(x),
                 Dense(11 * 11 * 64, 256, relu),
                 Dropout(0.3),
                 Dense(256, 1, sigmoid)) |> gpu;
end


function get_cdcgan_generator(args)
    # This is just the generator proposed in the Keras tutorial
    # https://github.com/malzantot/Pytorch-conditional-GANs
    latent_dim = args["latent_dim"]
    if args["activation"] in ["celu", "elu", "leakyrelu", "trelu"]
        # Now continue: We want to use Base.Fix2
        act = Fix2(getfield(NNlib, Symbol(args["activation"])), Float32(args["activation_alpha"]))
    else
        act = getfield(NNlib, Symbol(args["activation"]));
    end
    return Chain(Dense(latent_dim + 10, 20 * 20 * (latent_dim + 10), act),
                        x -> reshape(x, (20, 20, latent_dim + 10, :)),
                        ConvTranspose((5, 5), latent_dim + 10 => 32, bias=false), 
                        BatchNorm(32, act),
                        ConvTranspose((5, 5), 32 => 1, tanh, bias=false)) |> gpu;
end

# End of file models.jl
