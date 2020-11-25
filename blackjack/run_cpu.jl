include("cpu_blackjack.jl")

steps = Int(1e7)
a = Agent()

print("Starting play...")
play(a, steps)
print("Finished ", steps, " steps.")
