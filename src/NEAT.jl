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
    nodes_num::Int
    weights_num::Int
    generation::Int
end

# モデルを作る
function Model(input_size::Int, output_size::Int; T=Float32, rate=0.5, rng=MersenneTwister(0))
    nodes = []
    nodes_num = input_size + output_size
    weights_num = 0

    # InputNodeの設定
    for i in 1:input_size
        # 出力へつなげたり繋げなかったりする
        for i in (input_size+1):nodes_num
            if rand(rng) < rate 
                push!(targets, i)
                weights_num += 1
            end
        end
        node = InputNode(i, T(0), targets, 2rand(T, length(targets)).-1)
        push!(nodes, node)
    end

    # OutputNodeの設定
    for i in (input_size+1):nodes_num
        node = OutputNode(i, T(0), 2rand(T)-1, x -> x)
        push!(nodes, node)
    end

    return Model{T}(nodes, input_size, output_size, nodes_num, weights_num, 1)
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
    reset!(model)
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

mutable struct Species
    representative_id::Int
    nodes_num::Int
    weights_num::Int
    model_ids::Array{Int, 1}
end

mutable struct Population{T<:Number}
    models::Array{Model{T}, 1}
    species_set::Array{Species, 1}
    species_num::Int
    generation::Int
end


function node_distane(model1::Model, model2::Model)

end

function connection_distance(model1::Model, model2::Model)

end

# 種を分ける (最初の世代の場合)
function Population(models::Array{Model, 1}; compatibility_threshold=3.0<:AbstractFloat, compatibility_disjoint_coefficient=1.0<:AbstractFloat)
    species_set_toReturn = [Species(1, models[1].nodes_num, models[1].weights_num, [1])]
    species_set_toExamine = copy(species_set_toExamine)
    species_count = 0
    unexamined = [i for i in 1:length(models)]

    while length(species_set_toExamine) > 0
        current_species = pop!(species_set_toExamine)
        species_count += 1

        for i in unexamined
            distance = node_distance(models[current_species.representative_id], models[i]) + connection_distance(models[current_species.representative_id], models[i])
            if (distance > compatibility_threshold && length(species_set_toExamine) == 0)
                new_species = Species(i, models[i].nodes_num, models[i].weights_num, [i])
                push!(species_set_toReturn, new_species)
                push!(species_set_toExamine, new_species)
            elseif (distance <= compatibility_threshold)
                push!(species_set_toReturn.model_ids, i)
                delete!(unexamined, i)
            end
        end
    end

    return Population(models, species_set_toReturn, length(species_set_toReturn), models[1].generation)
end

# 種を分ける（最初の世代じゃない場合）
function Population(models::Array{Model, 1}, previous_population::Population; compatibility_threshold=3.0<:AbstractFloat, compatibility_disjoint_coefficient=1.0<:AbstractFloat)

end
