using Random
using Flux
using Flux: logitcrossentropy, onehot, onecold
using Plots

function multi_logistic(X, Y, reps)
    # data sizes
    n,d = size(X)
    m = size(Y,2)
    
    # linear predictor parameter
    theta = zeros(d,m)

    # predictor
    predicty(x) = theta'*x
    margin(pi, pj, y) = (2*dot(pi-pj,y) + dot(pj,pj) - dot(pi,pi)) / (2*norm(pi-pj) + 1e-10)
    multilogisticloss(yhat, y) = sum([exp(margin(r, y, yhat)) for r in reps])
    loss(x,y) = multilogisticloss(predicty(x), y)

    data = zip(eachrow(X), eachrow(Y))
    opt = ADAMW()
    Flux.@epochs 500 Flux.train!(loss, params(theta), data, opt)
    return predicty, theta
end