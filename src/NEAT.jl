using Random

abstract type AbstractEnv end
# もしくは `using ReinforcementLearningBase` するか (要検討)
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
    score::T
end

# モデルを作る
function Model(input_size::Int, output_size::Int; T=Float64, rate=0.5, rng=MersenneTwister())
    nodes = Array{Node, 1}()
    nodes_num = input_size + output_size
    weights_num = 0
    
    # InputNodeの設定
    for i in 1:input_size
        # 出力へつなげたり繋げなかったりする
        targets = Array{Int, 1}()
        for j in (input_size+1):nodes_num
            if rand(rng) < rate
                push!(targets, j)
                weights_num += 1
            end
        end
        node = InputNode(i, T(0), targets, 2rand(rng, T, length(targets)).-1)
        push!(nodes, node)
    end

    # OutputNodeの設定
    for i in (input_size+1):nodes_num
        node = OutputNode(i, T(0), 2rand(T)-1, x -> x)
        push!(nodes, node)
    end

    return Model{T}(nodes, input_size, output_size, nodes_num, weights_num, T(0))
end

Model{T}(input_size, output_size; kwargs...) where {T} = Model(input_size, output_size; T=T, kwargs...)

# InputNode以外の全ノードのstateをbiasにする
function reset!(model::Model)
    for i in (model.input_size+1):length(model.nodes)
        model.nodes[i].state = model.nodes[i].bias
    end
    nothing
end

# モデルに入力を入れて計算する
function (model::Model{T})(input::Array{T, 1}) where T <: Number
    reset!(model)
    middlenode_ids = Array{Int, 1}()
    
    # InputNodeからの出力を計算
    for i in 1:model.input_size
        model.nodes[i].state = input[i]
        for j in 1:length(model.nodes[i].targets)
            target = model.nodes[i].targets[j]
            model.nodes[target].state += model.nodes[i].weights[j] * model.nodes[i].state
            if typeof(model.nodes[j]) <: MiddleNode
                push!(target, middlenode_ids)
            end
        end
    end

    # MiddleNodeからの出力を計算
    while length(middlenode_ids) > 0
        id = popfirst!(middlenode_ids)
        for j in 1:length(model.nodes[id].targets)
            target = model.nodes[i].targets[j]
            model.nodes[target].state += model.nodes[id].activation_function(model.nodes[id].weights[j] * model.nodes[id].state)
        end
        if typeof(model.nodes[j]) == MiddleNode
            push!(j, middlenode_ids)
        end
    end

    return [model.nodes[i].activation_function(model.nodes[i].state) for i in (model.input_size+1):(model.input_size+model.output_size)]
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


function node_distance(model1::Model, model2::Model; compatibility_weight_coefficient=0.5, compatibility_disjoint_coefficient=0.5)
    disjoint_nodes = abs(model1.nodes_num - model2.nodes_num) * compatibility_disjoint_coefficient

    bias_distance = 0.0
    for node1 in model1.nodes
        if typeof(node1) <: InputNode
            continue
        end
        for node2 in model2.nodes
            if typeof(node2) <: InputNode
                continue
            end
            if node1.id == node2.id
                bias_distance += abs(node1.bias - node2.bias)
                break
            end
        end
    end
    bias_distance *= compatibility_weight_coefficient

    return (disjoint_nodes+bias_distance)/max(model1.nodes_num, model2.nodes_num)
end

function connection_distance(model1::Model, model2::Model; compatibility_weight_coefficient=0.5, compatibility_disjoint_coefficient=0.5)
    disjoint_connections = abs(model1.weights_num - model2.weights_num) * compatibility_disjoint_coefficient

    # nodeの数の6乗ぐらいの計算量のクソアルゴリズムなので改善したい
    weight_distance = 0.0
    for node1 in model1.nodes
        if typeof(node1) <: OutputNode
            continue
        end
        for node2 in model2.nodes
            if typeof(node2) <: OutputNode
                continue
            end
            if node1.id == node2.id

                for i in 1:length(node1.targets)
                    target1 = node1.targets[i]
                    for j in 1:length(node2.targets)
                        target2 = node2.targets[j]
                        if target1 == target2
                            weight_distance += abs(node1.weights[i] - node2.weights[j])
                            break
                        end
                    end
                end
                break

            end
        end
    end
    weight_distance *= compatibility_weight_coefficient

    return (disjoint_connections + weight_distance) / max(model1.weights_num, model2.weights_num)
end

# 種を分ける (最初の世代の場合)
function Population(models::Array{Model{T}, 1}; compatibility_threshold=3.0, compatibility_disjoint_coefficient=0.5, compatibility_weight_coefficient=0.5) where T
    species_set_toReturn = [Species(1, models[1].nodes_num, models[1].weights_num, [1])]
    species_set_toExamine = copy(species_set_toReturn)
    unexamined = [i for i in 2:length(models)]

    while length(unexamined) > 0
        current_species = pop!(species_set_toExamine)
        for i in copy(unexamined)
            distance = node_distance(models[current_species.representative_id], models[i], 
                            compatibility_disjoint_coefficient=compatibility_disjoint_coefficient,
                            compatibility_weight_coefficient=compatibility_weight_coefficient) 
                        + connection_distance(models[current_species.representative_id], models[i],
                            compatibility_disjoint_coefficient=compatibility_disjoint_coefficient,
                            compatibility_weight_coefficient=compatibility_weight_coefficient)

            if (distance > compatibility_threshold && length(species_set_toExamine) == 0)
                new_species = Species(i, models[i].nodes_num, models[i].weights_num, [i])
                push!(species_set_toReturn, new_species)
                push!(species_set_toExamine, new_species)
                deleteat!(unexamined, findfirst(x->x==i, unexamined))
            elseif (distance <= compatibility_threshold)
                push!(species_set_toReturn[end].model_ids, i)
                deleteat!(unexamined, findfirst(x->x==i, unexamined))
            end
        end
    end

    return Population{T}(models, species_set_toReturn, length(species_set_toReturn), 1)
end

# 種を分ける（最初の世代じゃない場合）
function Population(models::Array{Model{T}, 1}, previous_population::Population; 
    compatibility_threshold=3.0, compatibility_disjoint_coefficient=1.0, compatibility_weight_coefficient=0.5) where T

    species_set_toReturn = Array{Species, 1}()
    previous_species_set_toExamine = copy(previous_population.species_set)
    unexamined = [i for i in 1:length(models)]

    # 前の世代の種に含まれるかどうか調査する
    for current_species in previous_species_set_toExamine
        isnew = true

        for i in unexamined
            distance = node_distance(previous_population.models[current_species.representative_id], models[i], 
                            compatibility_disjoint_coefficient=compatibility_disjoint_coefficient,
                            compatibility_weight_coefficient=compatibility_weight_coefficient)
                        + connection_distance(previous_population.models[current_species.representative_id], models[i],
                            compatibility_disjoint_coefficient=compatibility_disjoint_coefficient,
                            compatibility_weight_coefficient=compatibility_weight_coefficient)

            if (distance <= compatibility_threshold && isnew)
                new_species = Species(i, models[i].nodes_num, models[i].weights_num, [i])
                push!(species_set_toReturn, new_species)
                deleteat!(unexamined, findfirst(x->x==i, unexamined))
                isnew = false
            elseif (distance <= compatibility_threshold)
                push!(species_set_toReturn[end].model_ids, i)
                deleteat!(unexamined, findfirst(x->x==i, unexamined))
            end
        end
    end

    # 新種を分ける
    species_set_toExamine = Array{Species, 1}()
    while length(unexamined) > 0
        new_species = Species(unexamined[1], models[unexamined[1]].nodes_num, models[unexamined[1]].weights_num, [unexamined[1]])
        popfirst!(unexamined)

        for i in unexamined
            distance = node_distance(models[new_species.representative_id], models[i], 
                            compatibility_disjoint_coefficient=compatibility_disjoint_coefficient,
                            compatibility_weight_coefficient=compatibility_weight_coefficient)
                        + connection_distance(models[new_species.representative_id], models[i],
                            compatibility_disjoint_coefficient=compatibility_disjoint_coefficient,
                            compatibility_weight_coefficient=compatibility_weight_coefficient)

            if (distance <= compatibility_threshold)
                push!(new_species.model_ids, i)
                deleteat!(unexamined, findfirst(x->x==i, unexamined))
            end
        end
        push!(species_set_toReturn, new_species)
    end

    return Population{T}(models, species_set_toReturn, length(species_set_toReturn), previous_population.generation+1)
end

# 成績を評価
function evaluate!(model::Model, env::AbstractEnv)
    while !get_terminal(env)
        env(model(get_state(env)))
        model.score += get_reward(env)
    end
    nothing
end

function evaluate!(popu::Population, env::AbstractEnv)
    bestmodel_id = popu.model_ids[1]

    for model in popu.models
        evaluate!(model, env)
        if popu.models[bestmodel_id].score < model.score
            bestmodel_id = model.id
        end
    end
    
    return bestmodel_id
end
