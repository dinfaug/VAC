struct RateTree   # makes new data type for tree
    data::Vector{Float64}
    rates_size::Tuple
    function RateTree(rates::AbstractArray) # function to produce tree
        n = 2^ndigits(length(rates) - 1, base=2) # honest nearest to the length of the rates vector
        data = zeros(2*n) # tree in 2 times longer then rates vector
        data[n : n+length(rates)-1] = vec(rates) # half of the tree is the rates vector
        while n > 1
            n >>= 1 #n/2
            for i in 0:n-1
                data[n + i] = data[2*(n + i)] + data[2*(n + i) + 1] # growштп a tree by summing the lower elements 
            end
        end
        return new(data, size(rates)) #return data (tree)  + rates_size
    end
end

function get_rates(tree::RateTree)
    n = length(tree.data) >> 1
    return reshape(tree.data[n : n+prod(tree.rates_size)-1], tree.rates_size)
    
end

function Base.setindex!(tree::RateTree, rate, inds...) # update tree function 
    n = (length(tree.data) >> 1) + LinearIndices(tree.rates_size)[CartesianIndex(inds)] - 1  #find number of element in the tree corresponding to the event
    tree.data[n] = rate # change it
    while n > 1
        s = tree.data[n] + tree.data[n ⊻ 1] # ⊻ is exclusive "or"
        n >>= 1
        tree.data[n] = s # change the upper elements 
    end
end

function treesearch(tree::RateTree, v) # search event index corresponding to the random number v
    k = 2
    # tree[3] is (rate[1] + ... + rate[n÷2])
    l = length(tree.data)
    while k < l
        if v > tree.data[k]
            v -= tree.data[k]
            k += 1
        end
        k <<= 1
    end
    return (k - l) >> 1 + 1, v # return the number of the event
end

function total_rate(tree::RateTree) # return total rates vector that is first element of the tree
    return tree.data[1]
end

function sample(tree::RateTree)  # produce index of the random event from the tree
    v = total_rate(tree)*rand() # random number
    i, dv = treesearch(tree, v) # find random event index in the tree
    return CartesianIndices(tree.rates_size)[i], dv # return cartesian coordinate of the random event 
end


function sample(trees::Dict{Symbol, RateTree})
    ks = collect(keys(trees))
    total_rates = total_rate.(values(trees))
    key = wsample(ks, total_rates)
    dt = 1/sum(total_rates)
    return dt, key, sample(trees[key])
end

# mini tree

function fill_mini_tree!(W::AbstractArray)
    W[7,..] = W[1,..] + W[2,..]
    W[8,..] = W[3,..] + W[4,..]
    W[9,..] = W[5,..] + W[6,..]
    W[10,..] = W[7,..] + W[8,..] + W[9,..]
end

function sample_hop(w::AbstractVector, dv::Float64)
    if dv < w[7]
        if dv < w[1]
            return 1
        else
            return 2
        end
    else
        dv -= w[7]
        if dv < w[8]
            if dv < w[3]
                return 3
            else
                return 4
            end
        else
            dv -= w[8]
            if dv < w[5]
                return 5
            else
                return 6
            end   
        end
    end
end

