using Random

abstract type Node end

mutable struct InputNode{T<:Number} <: Node
    id::Int
    state::T
    targets::Array{Int, 1}
    weights::Array{T, 1}
end

mutable struct MiddleNode{T<:Number} <: Node
    id::Int
    state::T
    targets::Array{Int, 1}
    weights::Array{T, 1}
    bias::T
    activation_function::Function
end

mutable struct OutputNode{T<:Number} <: Node
    id::Int
    state::T
    bias::T
    activation_function::Function
end

mutable struct Model{T<:Number}
    nodes::Array{Node, 1}
    input_size::Int
    output_size::Int
    node_num::Int
end

# モデルを作る
function Model(input_size::Int, output_size::Int; T=Float32, rng=MersenneTwister(0))
    nodes = []
    node_num = input_size + output_size

    # InputNodeの設定
    for i in 1:input_size
        targets = [i for i in (input_size+1):node_num]
        node = InputNode(i, T(0), targets, 2rand(T, output_size).-1)
        push!(nodes, node)
    end

    # OutputNodeの設定
    for i in (input_size+1):node_num
        node = OutputNode(i, T(0), 2rand(T)-1, x -> x)
        push!(nodes, node)
    end

    return Model{T}(nodes, input_size, output_size, node_num)
end

# 全ノードのstateをbiasにする
function reset!(model::Model)
    for i in 1:length(model.nodes)
        model.nodes[i].state = model.nodes[i].bias
    end
    nothing
end

# モデルに入力を入れて計算する
function (model::Model{T})(input::Array{T, 1}) where T <: Number
    middlenode_ids = []

    # InputNodeからの出力を計算
    for i in 1:model.input_size
        model.nodes[i].state = input[i]
        for j in 1:length(model.nodes[i].targets)
            target = model.nodes[i].targets[j]
            model.nodes[target].state += model.nodes[i].weights[j] * model.nodes[i].state
            if typeof(model.nodes[j]) == MiddleNode
                push!(target, middlenode_ids)
            end
        end
    end

    # MiddleNodeからの出力を計算
    while length(middlenode_ids) > 0
        id = popfirst!(middlenode_ids)
        for j in 1:length(model.nodes[id].targets)
            target = model.nodes[i].targets[j]
            model.nodes[target].state += model.nodes[id].weights[j] * model.nodes[id].state
        end
        if typeof(model.nodes[j]) == MiddleNode
            push!(j, middlenode_ids)
        end
    end

    return [model.nodes[i].state for i in (model.input_size+1):(model.input_size+model.output_size)]
end
