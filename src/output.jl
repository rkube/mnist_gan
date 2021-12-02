using Plots

# Use this function to generate 5 instances of target, the number.
# Then plot it in a heatmap. Transpose the output block-wise before plotting it

function gan_image_output(generator, target)
	n_samples = 5
	noise = randn(latent_dim, n_samples);
	random_labels = [target, target, target, target, target];
	random_labels_oh = Main.Flux.onehotbatch(random_labels, 0:9);
	@show random_labels
	random_vector_labels = cat(noise, random_labels_oh, dims=1);
	fake_img = reshape(g_cpu(random_vector_labels), 28, n_samples*28);
	
	fake_img = hcat([transpose(fake_img[:, k * 28:-1:(k-1) * 28 + 1]) for k in 1:n_samples]...)
	heatmap(fake_img, c=:grays, title=random_labels)
	p = heatmap(fake_img, c=:grays, title=random_labels)
	p
end

