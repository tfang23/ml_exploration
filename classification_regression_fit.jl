using Random
using Flux
using Flux: logitcrossentropy, onehot, onecold
using Plots

function classification_regression_fit(X, Y, l, r, lambda, kappa; numiters = 500)
    #     Inputs:
    #         X: input data
    #         Y: output data
    #         l: l(yhat, y)
    #         r: r(theta)
    #         lambda: regularization hyper-parameter
    #         numiters (optional): number of iterations
        
        data = zip(eachrow(X), eachrow(Y))
        n,d = size(X)
        theta = zeros(d)
        predicty(x) = theta'*x
        loss(x, y) = l(predicty(x), y[1], kappa) + lambda*r(theta)
        cost(x,y) = loss(x,y) + lambda*r(theta)
        risk() = sum((cost(d...) for d in data))/n
        opt = Flux.ADAGrad()
        losses = []
        tracker() = push!(losses, risk())
        Flux.@epochs numiters Flux.train!(loss, Flux.params(theta), data, opt, cb = Flux.throttle(tracker,10))
        return theta
    end