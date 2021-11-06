
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
    # First, let's make sure that we are working in Float32 here.
    α = Float32(args["activation_alpha"])
    act(x) = args["activation"] in ["leakyrelu", "elu", "celu"] ? getfield(Flux, Symbol(args["activation"]))(x, α) : getfield(Flux, Symbol(args["activation"]))(x)

    return Chain(Dense(28 * 28, 1024, relu),
                 Dropout(0.3),
                 Dense(1024, 512, relu),
                 Dropout(0.3),
                 Dense(512, 256, relu),
                 Dropout(0.3),
                 Dense(256, 1, sigmoid)) |> gpu
end


function get_vanilla_generator(args)
    α = Float32(args["activation_alpha"])
    act(x) = args["activation"] in ["leakyrelu", "elu", "celu"] ? getfield(Flux, Symbol(args["activation"]))(x, α) : getfield(Flux, Symbol(args["activation"]))(x)

    return Chain(Dense(args["latent_dim"], 256, relu),
                 Dense(256, 512, relu),
                 Dense(512, 1024, relu),
                 Dense(1024, 28*28, tanh)) |> gpu
end
