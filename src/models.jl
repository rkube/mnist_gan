
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
    # This is adapted from the Keras tutorial: https://keras.io/examples/generative/conditional_gan
    #if args["activation"] in ["celu", "elu", "leakyrelu", "trelu"]
    #    # Now continue: We want to use Base.Fix2
    #    act = Fix2(getfield(NNlib, Symbol(args["activation"])), Float32(args["activation_alpha"]))
    #else
    #    act = getfield(NNlib, Symbol(args["activation"]));
    #end
    act = Fix2(getfield(NNlib, Symbol("leakyrelu")), Float32(0.2));
    
    return Chain(Conv((3, 3), 11 => 64, stride=(2, 2), pad=SamePad(), act),
                 Conv((3, 3), 64 => 128, stride=(2, 2), pad=SamePad(), act),
                 GlobalMaxPool(),
                 x -> flatten(x),
                 Dense(128, 1, sigmoid))
end


function get_cdcgan_generator(args)
    # This is just the generator proposed in the Keras tutorial
    # https://keras.io/examples/generative/conditional_gan
    act = Fix2(getfield(NNlib, Symbol("leakyrelu")), Float32(0.2));
    return Chain(Dense(110, 7*7*110, act),
                 x -> reshape(x, (7, 7, 110, :)),
                 ConvTranspose((4, 4), 110 => 128, stride=(2, 2), pad=SamePad(), act),
                 ConvTranspose((4, 4), 128 => 128, stride=(2, 2), pad=SamePad(), act),
                 Conv((7, 7), 128 => 1, pad=SamePad()))

end





