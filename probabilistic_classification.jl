using MLDatasets, Flux, Distributions, DataFrames

function probabilistic_classification(X, Y; numiters=500)
    # data sizes
    n,d = size(X)
    K = size(Y, 2)

    W = randn(K, d)
    b = zeros(K)

    model = x -> Flux.softmax(W * x .+ b)

    opt = Descent(0.1)

    loss = (x, y) -> Flux.crossentropy(model(x), y)

    data = zip(eachrow(X), eachrow(Y))
    opt = ADAMW()
    Flux.@epochs 500 Flux.train!(loss, Flux.params(W, b), data, opt)
    return model
end