# dependencies
using ResumableFunctions
using ConcurrentSim
using Logging
using Distributions

# ------------------------------ #
# Resumable functions demo
# ------------------------------ #
@resumable function pascal()
    # two initial values
    a = [1]
    @yield a
    a = [1,1]
    @yield a
    # all the following values
    while true
        a = vcat([1],a[1:end-1] .+ a[2:end],[1])
        @yield a
    end
end


@resumable function findroot(x0::Number)
    # note: based on Newton's method
    res = x0
    @yield res
    while true
        res = res - (res^2 - x0) / (2*res)
        @yield res
    end  
end

# ------------------------------ #
# Container demo
# ------------------------------ #

@resumable function fill(sim::Simulation, c::Container)
    while true 
        @yield timeout(sim, rand(1:10))
        @yield put(c,1)
        @info "$(now(sim)) - item added to the container"
    end
end

@resumable function empty(sim::Simulation, c::Container)
    while true
        @yield timeout(sim, rand(1:10))
        n = rand(1:3)
        @info "$(now(sim)) - Filed a request for $(n) items"
        @yield get(c,n)
        @info "$(now(sim)) - Got my $(n) items"
    end
end

@resumable function monitor(sim::Simulation, c::Container)
    while true
        @info "$(now(sim)) - current container level: $(c.level)/$(c.capacity)"
        @yield timeout(sim, 1)
    end
end
	
function containerdemo()
	# setup the simulation
	@info "\n$("-"^70)\nWorking with containers\n$("-"^70)\n"
	sim = Simulation()
	c = Container(sim,10)
	@process fill(sim,c)
	@process monitor(sim,c)
	@process empty(sim,c)
	run(sim,30)
end

# ------------------------------ #
# Store demo
# ------------------------------ #

# our own type of object
struct Object
    id::Int
end

@resumable function fill(sim::Simulation, s::Store)
    i = 0
    while true 
        i += 1
        item = Object(i)
        @yield timeout(sim, rand(1:10))
        @yield put(s,item)
        @info "$(now(sim)) - item $(item) added to the store"
    end
end

@resumable function empty(sim::Simulation, s::Store)
    while true
        @yield timeout(sim, rand(1:10))
        n = rand(1:3)
        @info "$(now(sim)) - filed my request for $(n) items"
        for _ in 1:n
            @yield get(s)
        end
        @info "$(now(sim)) - Got my $(n) items"
    end
end

@resumable function monitor(sim::Simulation, s::Store)
    while true
        @info "$(now(sim)) - current store level: $(length(s.items))/$(s.capacity)"
        @yield timeout(sim, 1)
    end
end

function storedemo() 
    # setup the simulation
    @info "\n$("-"^70)\nWorking with stores\n$("-"^70)\n"
    sim = Simulation()
    s = Store{Object}(sim, capacity=UInt(10))

    @process fill(sim, s)
    @process empty(sim, s)
    @process monitor(sim, s)
    run(sim,30)
end

# ------------------------------ #
# Dependency demo
# ------------------------------ #
@resumable function basic(sim::Simulation)
    @info "$(now(sim)) - Basic goes to work"
    p = @process bottleneck(sim)
    @yield p
    @info "$(now(sim)) - Basic continues after bottleneck completion"
end

@resumable function bottleneck(sim::Simulation)
       @yield timeout(sim, 10)
end

function dependencydemo()
    @info "\n$("-"^70)\nProcess dependencies\n$("-"^70)\n"
    sim = Simulation()
    @process basic(sim)
    run(sim)
end

# ------------------------------ #
# Conditional execution demo
# ------------------------------ #
@resumable function agent(env::Environment,r::Resource)
    req = request(r)
    res = @yield req | timeout(env, 4)
    if res[req].state == ConcurrentSim.processed
        @info "$(env.time) - Agent is using the resource..."
        @yield timeout(env,1)
        release(r)
        @info "$(env.time) - Agent released the resource."
    else
        @info "$(env.time) - Patience ran out..."
        cancel(r, req)
    end
end

function conditional_execution_demo()
    @info "\n$("-"^70)\nOne of both events\n$("-"^70)\n"
    sim = Simulation()
    r = Resource(sim,0)
    @process agent(sim, r)
    run(sim,10)
end