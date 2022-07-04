include("houseplots.jl")

using CSV
using Random
using Statistics
using LinearAlgebra

stringtonumber(x) = parse(Float64, x)
getdatafield(D, name) = hcat([getproperty(row, Symbol(name)) for row in D])
normsquared(x) = sum(x.*x)
rmse(yhat, y) = sqrt(normsquared(yhat-y)/length(y))

function csv(fname)
    df=CSV.File(fname, type=String, header=1)
    # The df is a CSV.File object
    header = string.(propertynames(df))
    return df, header
end

function loaddata()
    D, header = csv("house.csv")
    println(header)
    notoutliers = [row for row in D if parse(Float64, row[:GrLivArea]) <= 4000]
    D2 = vcat(notoutliers...)
    return D2, header
end


function randomsplit(n, trainfrac=0.8)
    ntrain = convert(Int64, round(trainfrac*n))
    p = Random.randperm(n)
    trainrows = sort(p[1:ntrain])
    testrows = sort(p[ntrain+1:n])
    return trainrows, testrows
end

function applysplit(X, Y, trainrows, testrows)
    return X[trainrows,:], Y[trainrows,:], X[testrows,:],  Y[testrows,:]
end

function getstatistics(U)
    means = [Statistics.mean(x) for x in eachcol(U)]
    stds  = [Statistics.std(x) for x in eachcol(U)]
    return means, stds
end

function standardizeplusone(X,means,stds)
    Z = zeros(size(X))
    for i=1:size(X,2)
        if stds[i] != 0
            Z[:,i] = (X[:,i] .- means[i])/stds[i]
        else
            Z[:,i] = X[:,i] .- means[i]
        end
    end
    n = size(X,1)
    Z = [ones(n,1) Z]
    return Z
end
    

function embedx(D)
    field(name) = getdatafield(D, name)
    realf(name) = stringtonumber.(field(name))
    X = hcat(realf("YearBuilt"),    # numeric
             realf("GrLivArea"),    # numeric
             realf("1stFlrSF"),     # numeric
             realf("2ndFlrSF"),     # numeric, many 0
             realf("GarageArea"),   # numeric, many 0 
             realf("WoodDeckSF"),   # numeric, many 0 
             realf("TotalBsmtSF"),  # numeric, many 0
             realf("YearRemodAdd"), # numeric, many at 1950
             realf("LotArea"),      # numeric, with some outliers
             realf("BedroomAbvGr"), # ordinal 0-8
             realf("KitchenAbvGr"), # ordinal 0-3
             realf("Fireplaces"),   # ordinal 0-3
             realf("HalfBath"),     # ordinal 0-2
             realf("TotRmsAbvGrd"), # ordinal 2-14
             realf("OverallCond"),  # ordinal 1-9
             realf("OverallQual"),  # ordinal 1-10
             realf("GarageCars"),   # ordinal 0-4
             unlikert.(field("KitchenQual")),  # ordinal, but "Ex", "Gd", "TA", "Fa", "Po"
             onehot(field("Neighborhood")),   # 25 different names
             onehot(field("BldgType")),       # 5 different types
           )
    return X
end


function unlikert(s)
    d = Dict("Ex" =>5, "Gd" =>4, "TA" => 3, "Fa" => 2, "Po" => 1)
    return d[s]
end


embedy(D) = log.(stringtonumber.(getdatafield(D, :SalePrice)))


function onehot(u)
    categories = unique(u)
    catnum(s) = findfirst(x -> x==s, categories)
    n = length(u)
    K = length(categories)
    Y = zeros(n,K)
    for i=1:n
        c = catnum(u[i])
        Y[i,c] = 1
    end
    return Y
end

function ridgeregressionconstfeature(X,Y,lambda)  
    n,d = size(X)
    m = size(Y,2)
    E = [zeros(d-1,1)  I(d-1)]
    A = [X; sqrt(lambda*n)*E]
    B = [Y; zeros(d-1,m)]
    theta = A\B
end



function main()
    D, header = loaddata()
    n = size(D,1)
    Y = embedy(D)
    X0 = embedx(D)
    trainrows, testrows = randomsplit(n)
    Xtrain0, Ytrain, Xtest0, Ytest = applysplit(X0, Y, trainrows, testrows)
    means, stds = getstatistics(Xtrain0)
    Xtrain = standardizeplusone(Xtrain0, means, stds)
    Xtest  = standardizeplusone(Xtest0, means, stds)
    lambdas = 10 .^ range(-3,3,length=50)
    thetas = [ridgeregressionconstfeature(Xtrain, Ytrain, lambda) for lambda in lambdas]
    train_errors = [rmse(Xtrain*theta, Ytrain) for theta in thetas]
    test_errors  = [rmse(Xtest*theta, Ytest) for theta in thetas]
    testmin, imin = findmin(test_errors)
    lambdamin = lambdas[imin]
    thetamin = thetas[imin]
    
    println("optimal train rmse = ", minimum(train_errors))
    println("optimal test rmse = ", minimum(test_errors))
    println("optimal lambda = ", lambdamin)
    println("optimal theta:")
    display(thetamin)

    houseplots(D, header, thetas, train_errors, test_errors,
               lambdas, Xtest, Y, Ytest, thetamin)
end

main()
