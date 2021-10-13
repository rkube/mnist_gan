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
η = 1e-3

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





end # module
