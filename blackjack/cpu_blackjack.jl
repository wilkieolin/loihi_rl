n_actions = 2
dealer_range = 1:10
player_range = 12:21

mutable struct State
    usable_ace::Int
    dealer_card::Int
    player_sum::Int
    
    #random starts
    State() = new(rand(1:2), rand(dealer_range), rand(player_range))
end

mutable struct Rewards
    pos::Array{Int,4}
    neg::Array{Int,4}
    
    Rewards() = new(ones(n_actions, 2, length(dealer_range), length(player_range)), 
                    ones(n_actions, 2, length(dealer_range), length(player_range)))
end

mutable struct Agent
    state::State
    rewards::Rewards
    
    Agent() = new(State(), Rewards())
end

function draw_card()
    card = rand(1:12)
    #all face cards count as 10
    if card > 10 card = 10 end
    return card
end

function randomize_state!(state::State)
    state.usable_ace = rand(1:2)
    state.dealer_card = rand(dealer_range)
    state.player_sum = rand(player_range)
end

function get_action(player::Agent)
    state = player.state
    view = x -> x[:, state.usable_ace, state.dealer_card, state.player_sum - first(player_range) + 1]
    
    state_action_rewards = view(player.rewards.pos)
    state_action_punishments = view(player.rewards.neg)
    
    values = state_action_rewards ./ (state_action_punishments .+ state_action_rewards)
    #greedy policy
    max = maximum(values)
    #break evenly between ties
    matches = findall(values .== max)
    if length(matches) > 1
        action = rand(matches)
    else
        action = matches[1]
    end
    
    return action
end

function advance(player::Agent, step::Int)
    state = player.state
    action = Int(0)
    reward = Int(0)
    
    #use random starts
    if step == 0
        action = rand(1:n_actions)
    else
        action = get_action(player)
    end
    
    #hit = 1
    if action == 1
        next_state = deepcopy(player.state)
        next_state.player_sum += draw_card()
        
        if next_state.player_sum > 21 && next_state.usable_ace == 2
            #mark our ace as unusable since it will make us go bust
            next_state.player_sum -= 10
            next_state.usable_ace -= 1
        elseif next_state.player_sum > 21
            #player has gone bust
            next_state = State()
            reward = -1
        else
            reward = 0
        end
    #stick = 2
    else
        state.dealer_card == 1 ? dealer_ace = true : dealer_ace = false
        dealer_sum = state.dealer_card
        #next state will be a new game
        next_state = State()
        
        while (dealer_sum < 17)
            dealer_sum += draw_card()

            if (dealer_ace && dealer_sum > 21)
                #mark the dealer's ace as unusable
                dealer_sum -= 10
                dealer_ace = false
            end
        end
        
        #see if the dealer has gone bust, otherwise see who's closer
        if (dealer_sum > 21)
            reward = 1
        else
            gap = abs(dealer_sum - 21) - abs(state.player_sum - 21)
            if gap < 0
                #dealer closer
                reward = -1
            elseif gap == 0
                #draw
                reward = 2
            else
                #win on stick
                reward = 1
            end
        end
    end
    
    return (action, reward, next_state)
end

function isequal(s1::State, s2::State)
    if s1.usable_ace == s2.usable_ace && s1.dealer_card == s2.dealer_card && s1.player_sum == s2.player_sum
        return true
    else
        return false
    end
end

function play(player::Agent, steps::Int, verbose::Bool=false)
    step = Int(0)
    pairs = []
    rewards = zeros(Int, steps)
    actions = zeros(Int, steps)
    
    for i in 1:steps
        state = player.state
        
        if verbose println(string("UA ", state.usable_ace, " DC ", state.dealer_card, " PS ", state.player_sum)) end
        actions[i], rewards[i], next_state = advance(player, step)
        append!(pairs, [(actions[i], state)])
        
        if verbose println("Action ", actions[i]) end
        if verbose println("Reward ", rewards[i]) end
        
        if rewards[i] != 0
            #store the outcome
            for (a,s) in pairs
                increment = x -> x[a, s.usable_ace, s.dealer_card, s.player_sum - first(player_range)+1] += 1

                if rewards[i] > 0
                    increment(player.rewards.pos)
                else
                    increment(player.rewards.neg)
                end
            end
            
            empty!(pairs)
            step = 0
        else
            step += 1
        end
        
        player.state = next_state
    end
    
    return actions, rewards
end

function view_policy(player::Agent)
    values = player.rewards.pos ./ (player.rewards.neg .+ player.rewards.pos)
    return values[2,:,:,:] .- values[1,:,:,:]
end

