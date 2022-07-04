
import PyCall
import PyPlot; const plt = PyPlot
plt.plt.style.use("seaborn")


filename() = "house3"

# needs a function filename()
function plotname(x)
    a=findlast(isequal('.'), x)
    basename = x[1:a-1]
    extension = x[a+1:end]
    basename2 = replace(basename, '.' => "__")
    fname = filename()
    return  "graphics/$(fname)_$(basename2).$(extension)"
end

function save(name)
    if length(name)==0
        return
    end
    plt.savefig(plotname(name))
end

function plotridge(thetas, train_rmse, test_rmse, lambdas, name)
    println("$(name): optimal train rmse = ", minimum(train_rmse))
    println("$(name): optimal test rmse = ", minimum(test_rmse))
    plotregpath(train_rmse, test_rmse, lambdas, name)
    plotregpaththeta(thetas, lambdas, name)
end

function plotregpath(trainlosses, testlosses, lambdas, name)
    # error vs lambda
    fig, ax = plt.subplots(figsize=(4,3))
    ax.semilogx(lambdas, trainlosses,  markersize=0, linewidth=1, color="blue", alpha=1)
    ax.semilogx(lambdas, testlosses,  markersize=0, linewidth=1, color="red", alpha=1)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    ax.legend(["train", "test"])
    save("$(name)_regpath.pdf")
end

function plotregpaththeta(thetas, lambdas, name)
    # theta vs lambda
    fig, ax = plt.subplots(figsize=(4,3))
    T = hcat(thetas...)
    ax.semilogx(lambdas, T[2:end,:]', linewidth=2, markersize=0, alpha=1)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    save("$(name)_regtheta.pdf")
end


function plotfeature(u, Y, name)
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(u, Y, linewidth=0, markersize=3, marker="o")
    save("$(lowercase(name)).pdf")
end


function plotperf(y, yhat)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(y, yhat, linewidth=0, markersize=3, marker="o", color="red")
    plt.gca().set_aspect("equal", adjustable="box")
    yt = range(minimum(y), maximum(y), length=50)
    ax.plot(yt, yt, linewidth=1, markersize=0, color="black")
    save("perf.pdf")
end


function plottheta(theta)
    d = length(theta)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.bar(2:d,theta[2:end])
    save("thetabar.pdf")
end


function houseplots(D, header, thetas, train_errors, test_errors, lambdas, Xtest, Y, Ytest, thetamin)
    close("all")
    plotridge(thetas, train_errors, test_errors, lambdas, "ridge")
    plotfeature(stringtonumber.(getdatafield(D, "LotArea")), Y, "LotArea")
    plotfeature(stringtonumber.(getdatafield(D, "GrLivArea")), Y, "GrLivArea")
    plotperf(Ytest, Xtest*thetamin)
    plottheta(thetamin)
end
