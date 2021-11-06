
"""
Contains functions that return a model.


"""

function get_vanilla_discriminator(args)
    if args["activation"] in ["leakyrelu", "elu", "celu"]
        f = getfield(Flux, Symbol(args["activation"]))
        act(x) = f(x, args["activation_alpha"])
    else
        act(x) = getfield(Flux, Symbol(args["activation"]))(x)
    end

    return Chain(Dense(n_features, 1024, x -> act(x)),
                 Dropout(0.3),
                 Dense(1024, 512, x -> act(x)),
                 Dropout(0.3),
                 Dense(512, 256, x -> act(x)),
                 Dropout(0.3),
                 Dense(256, 1, sigmoid)) |> gpu
end


function get_vanilla_generator(args)
    if args["activation"] in ["leakyrelu", "elu", "celu"]
        f = getfield(Flux, Symbol(args["activation"]))
        act(x) = f(x, args["activation_alpha"])
    else
        act(x) = getfield(Flux, Symbol(args["activation"]))(x)
    end

    return Chain(Dense(args["latent_dim"], 256, x -> act(x)),
                 Dense(256, 512, x -> act(x)),
                 Dense(512, 1024, x -> act(x)),
                 Dense(1024, n_features, tanh)) |> gpu
end
