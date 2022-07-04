using Flux
using Random

# Random.seed!(0)

function nn_regression(X, Y, lambda; numiters=40)
    d = size(X,2); m = size(Y,2)

    model = Chain(
        Dense(d, 10, relu),
        Dense(10, 10, relu),
        Dense(10, 10, relu),
        Dense(10, m, identity))

    data = zip(eachrow(X), eachrow(Y))
    
    # Now define functions to pass to Flux.train!
    
    reg() = sum([norm(model[i].W, 2).^2 for i = 1:length(model)]) # Regularizer
    
    loss(x,y) = norm(model(x)-y, 2).^2
    cost(x,y) = loss(x,y) + lambda*reg()
    
    opt = Descent(0.001) # 0.001 is the learning rate

    # These lines all handle the callback which prints the loss 
    # Be careful with this RMSE function - it takes (yhat - y).^2 as one parameter
    function RMSE(rsquared)
        mse = sum(rsquared)/length(rsquared)
        return sqrt(mse)
    end

    ctr = 0    
    function callback()
        if ctr % 1000 == 0 # controls the frequency of printing the loss
            println("Loss: $(RMSE([loss(x,y) for (x,y) in data]))")
        end
        ctr += 1
    end
    # Done with callback

    # This line trains the model
    Flux.@epochs numiters Flux.train!(cost, Flux.params(model), data, opt, cb=callback)
    return model
end