
"""
Contains functions that return a model.


"""

function get_vanilla_discriminator()
    return Chain(Dense(n_features, 1024, x -> leakyrelu(x, 0.2f0)),
                 Dropout(0.3),
                 Dense(1024, 512, x -> leakyrelu(x, 0.2f0)),
                 Dropout(0.3),
                 Dense(512, 256, x -> leakyrelu(x, 0.2f0)),
                 Dropout(0.3),
                 Dense(256, 1, sigmoid)) #|> gpu
end


function get_vanilla_generator()
    return Chain(Dense(latent_dim, 256, x -> leakyrelu(x, 0.2f0)),
                 Dense(256, 512, x -> leakyrelu(x, 0.2f0)),
                 Dense(512, 1024, x -> leakyrelu(x, 0.2f0)),
                 Dense(1024, n_features, tanh)) #|> gpu
end
