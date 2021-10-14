module mnist_gan


# This is an implementation of a GAN to model the MNIST dataset
# Adapted from pytorch:
# https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f

using MLDatasets: MNIST
using Flux.Data: DataLoader
using Flux

batch_size=128
n_features = 28*28
n_gen = 100
num_epochs = 1
η = 2e-4

# Load MNIST train and test data
train_x, train_y = MNIST.traindata(Float32)
test_x, test_y = MNIST.testdata(Float32)

train_x = reshape(train_x, 28, 28, 1, :)
test_x = reshape(test_x, 28, 28, 1, :)

train_y, test_y = Flux.onehotbatch(train_y, 0:9), Flux.onehotbatch(test_y, 0:9)

# Now load the train images and labels into a DataLoader
train_loader = DataLoader((data=train_x, label=train_y), batchsize=batch_size, shuffle=true)

# Define the discriminator network. 
# The networks takes a flattened 28x28=784 image as input and outputs the
# probability of the image belonging to the real dataset.
train_x, train_y = MNIST.traindata(Float32)
test_x, test_y = MNIST.testdata(Float32)

discriminator = Chain(Dense(n_features, 1024, leakyrelu),
                      Dropout(0.3),
                      Dense(1024, 512, leakyrelu),
                      Dropout(0.3),
                      Dense(512, 256, leakyrelu),
                      Dropout(0.3),
                      Dense(256, 1, sigmoid))

# The input to the discriminator is a flat vector.
# To pass a batch of images to the discriminator we need to flatten the input
x, y = first(train_loader);
predictions = discriminator(flatten(x))
size(predictions) == (1, batch_size)


# The generator will generate images which come from the learned
# distribution. The output layer has a tanh activation function
# which maps the output to [-1:1], the same range as in the 
# pre-processed MNIST images

generator = Chain(Dense(n_gen, 256, leakyrelu),
                  Dense(256, 512, leakyrelu),
                  Dense(512, 1024, leakyrelu),
                  Dense(1024, n_features))

# The algorithm that describes training of a GAN using stochastic gradient descent
#
# for number in training iterations do
#   for k steps do
#       Sample minibatch of m noise examples z¹, …, zᵐ from noise prior pg(z)
#       Sample minibatch of m examples x¹, …, xᵐ from data generating distribution p_data(x)
#       Update the discriminator by ascending its stochastic gradient
#       ∇_theta_d 1\m Σ_{i=1}^{m} [ log D(xⁱ) + log(1 - D(G(zⁱ))]
#   end for
#   
#   Sample minibatch of m noise examples z¹, …, zᵐ from noise prior pg(z)
#   Update the generator by descending its stochastic gradient
#       ∇_theta_g 1/m Σ_{i=1}^{m} log(1 - D(G(zⁱ))
#   end for

# Optimizer for the discriminator
opt_dscr = ADAM(η)
opt_gen = ADAM(η)

function train_dscr(discriminator, real_data, fake_data)
    # Given real and fake data, update the parameters of the discriminator network
    # Assume that size(real_data) = 784xBS
    # Feed real data to the discriminator. For these, the discriminator should return ones.
    pred_real = discriminator(real_data)
    grads_real = gradient(Flux.params(discriminator)) do
        loss_real = Flux.Losses.logitbinarycrossentropy(pred_real, ones(size(pred_real)))
    end
    loss_real = Flux.Losses.logitbinarycrossentropy(pred_real, ones(size(pred_real)))
    

    # Train on fake data
    # Generate fake data and feed it to the discrminiator. For these, the discriminator should return zeros.
    pred_fake = discriminator(fake_data)
    grads_fake = gradient(Flux.params(discriminator)) do
        loss_fake = Flux.Losses.logitbinarycrossentropy(pred_fake, zeros(size(pred_fake)))
    end
    loss_fake = Flux.Losses.logitbinarycrossentropy(pred_fake, zeros(size(pred_fake)))

    # Update the parameters of the discriminator with the respective gradients.
    # This happens in-place(! of function signature)
    Flux.update!(opt_dscr, Flux.params(discriminator), grads_real)
    Flux.update!(opt_dscr, Flux.params(discriminator), grads_fake)

    return loss_real + loss_fake, pred_real, pred_fake
end


function train_gen(discriminator, generator)
    # Let the generator create fake data which should out-smart the discriminator
    # Let's see how well the discriminator does
    grads = gradient(Flux.params(generator)) do
        fake_data = generator(randn(n_gen, batch_size))
        pred = discriminator(fake_data)
        loss = Flux.Losses.logitbinarycrossentropy(pred, ones(size(pred)))
    end
    @show grads
    Flux.update!(opt_gen, Flux.params(generator), grads)
    # Return the loss
    loss = Flux.Losses.logitbinarycrossentropy(pred, ones(size(pred)))
end


for n in 1:num_epochs
    for (x, y) in train_loader
        # Train the discriminator
        # First flatten the images, which squashes all dimensions and keeps the 
        # the last dimension, which is the batch dimension
        x = flatten(x) 

        # Now generator fake data
        fake_data = generator(randn(n_gen, batch_size))

        # And train the discriminator
        loss_dscr, pred_real, pred_fake = train_dscr(discriminator, x, fake_data)

        # Train the generator
        loss_gen = train_gen(discriminator, generator)

        @show loss_dscr, loss_gen, pred_real, pred_fake
    end
end




