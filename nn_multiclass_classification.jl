function nn_multiclass_classification(X, Y, n_classes; numiters=40)

    d = size(X,2); m = size(Y,2)

    # Feel free to play with this model, add layers, change layer size
    # change activation, etc.
    model = Chain(
        Dense(d, 2*d, relu),
        Dense(2*d, n_classes, sigmoid))
    # The model outputs n_classes probabilities. We will choose the highest as our classification

    data = zip(eachrow(X), eachrow(Y))
	
    # logitcrossentropy is used for training classification models
    loss(x, y) = logitcrossentropy(model(x), y)
	
    # Training
    # Gradient descent optimiser with learning rate 0.5
    optimiser = Descent(0.5)

    # These lines all handle the callback which prints the loss
    ctr = 0    
    function callback()
        if ctr % 128 == 0 # controls the frequency of printing the loss
            println("Loss: $(sum([loss(x,y) for (x,y) in data]))")
        end
        ctr += 1
    end
    # Done with callback
    println("Starting training.")
    
    #Flux.train!(loss, params(model), train_data, optimiser)
	Flux.@epochs numiters Flux.train!(loss, Flux.params(model), data, optimiser, cb=callback)

    return model
end