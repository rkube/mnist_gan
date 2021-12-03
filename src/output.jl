using Plots
# Use this function to generate instances of target 
# Then plot it in a heatmap. Transpose the output block-wise before plotting it

function gan_image_output(generator, filename, args)
	n_digits = 10 # Number of digits
	n_samples = 10 # Number of samples per digits
    
    noise = randn(args["latent_dim"], n_digits * n_samples);
	random_labels = repeat(0:9, inner=n_samples);
	random_labels_oh = Main.Flux.onehotbatch(random_labels, 0:9);
	
	random_vector_labels = cat(noise, random_labels_oh, dims=1) |> gpu;
    fake_img = generator(random_vector_labels) |> cpu;
	fake_img = vcat([
		hcat([transpose(fake_img[:, 28:-1:1, 1, n_samples * (d - 1) + k]) for k in 1:n_samples]...)
		for d in 1:n_digits]...);
    savefig(heatmap(fake_img, c=:grays, title=random_labels), filename)
    return nothing
end

