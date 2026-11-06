include("runtests_setup.jl")

using Oceananigans.TimeSteppers: update_state!
using NumericalEarth.Radiations: PrescribedRadiation,
                                 SurfaceRadiationProperties,
                                 InterfaceRadiationFlux
using ClimaSeaIce.SeaIceThermodynamics: SeaIceColumnDiscretization,
                                       FixedDrainedIceSalinityProfile, melting_temperature

@testset "PrescribedRadiation construction" begin
    for arch in test_architectures
        A = typeof(arch)

        # Form B: grid-only constructor (zero downwelling, surface properties only)
        @info "Testing PrescribedRadiation(grid) on $A..."
        grid = RectilinearGrid(arch, size = 10, z = (-100, 0), topology = (Flat, Flat, Bounded))
        rad = PrescribedRadiation(grid)
        @test rad.surface_properties isa NamedTuple
        @test haskey(rad.surface_properties, :ocean)
        @test haskey(rad.surface_properties, :sea_ice)
        @test rad.surface_properties.ocean isa SurfaceRadiationProperties
        @test rad.surface_properties.ocean.albedo == 0.05
        @test rad.surface_properties.ocean.emissivity == 0.97
        @test rad.surface_properties.sea_ice.albedo == 0.7
        @test rad.surface_properties.sea_ice.emissivity == 1.0
        @test rad.stefan_boltzmann_constant ≈ 5.67e-8 atol=1e-10
        @test isnothing(rad.interface_fluxes)

        # Surfaces can be omitted
        @info "Testing PrescribedRadiation(grid; sea_ice_surface=nothing) on $A..."
        rad_ocean_only = PrescribedRadiation(grid; sea_ice_surface = nothing)
        @test haskey(rad_ocean_only.surface_properties, :ocean)
        @test !haskey(rad_ocean_only.surface_properties, :sea_ice)

        # Custom surface properties
        custom_ocean = SurfaceRadiationProperties(0.1, 0.95)
        rad_custom = PrescribedRadiation(grid; ocean_surface = custom_ocean)
        @test rad_custom.surface_properties.ocean.albedo == 0.1
        @test rad_custom.surface_properties.ocean.emissivity == 0.95

        # time_step! works
        @info "Testing time_step!(::PrescribedRadiation) on $A..."
        rad2 = PrescribedRadiation(grid)
        time_step!(rad2, 60.0)
        @test rad2.clock.time == 60.0
    end
end

@testset "PrescribedRadiation set!" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch, size = 1, z = (-1, 0), topology = (Flat, Flat, Bounded))
        rad = PrescribedRadiation(grid)

        set!(rad; downwelling_shortwave = 250, downwelling_longwave = 350)
        @test only(Array(interior(rad.downwelling_shortwave[1]))) == 250
        @test only(Array(interior(rad.downwelling_longwave[1])))  == 350

        # An omitted keyword leaves that field untouched.
        set!(rad; downwelling_shortwave = 100)
        @test only(Array(interior(rad.downwelling_shortwave[1]))) == 100
        @test only(Array(interior(rad.downwelling_longwave[1])))  == 350
    end
end

@testset "Radiation surface property providers" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch, size = 10, z = (-100, 0), topology = (Flat, Flat, Bounded))

        albedo = 0.15
        emissivity = 0.93
        land_surface = SurfaceRadiationProperties(; albedo, emissivity)

        rad = PrescribedRadiation(grid; ocean_surface = nothing,
                                        sea_ice_surface = nothing,
                                        land_surface)

        @test rad.surface_properties.land.albedo isa Number
        @test rad.surface_properties.land.emissivity isa Number
        @test NumericalEarth.stateindex(rad.surface_properties.land.albedo, 1, 1, 1, grid, 0, nothing) == 0.15
        @test NumericalEarth.stateindex(rad.surface_properties.land.emissivity, 1, 1, 1, grid, 0, nothing) == 0.93
    end
end

@testset "PrescribedRadiation paired with model" begin
    for arch in test_architectures
        A = typeof(arch)

        @info "Testing OceanOnlyModel with PrescribedRadiation on $A..."

        grid = RectilinearGrid(arch, size = 10, z = (-100, 0), topology = (Flat, Flat, Bounded))
        ocean = ocean_simulation(grid)
        radiation = PrescribedRadiation(grid)
        model = OceanOnlyModel(ocean; radiation)

        # interface_fluxes are allocated for present surfaces (ocean + sea_ice
        # via FreezingLimitedOceanTemperature).
        @test model.radiation.interface_fluxes.ocean isa InterfaceRadiationFlux
        @test model.radiation.interface_fluxes.sea_ice isa InterfaceRadiationFlux

        set!(radiation; downwelling_shortwave = 250, downwelling_longwave = 350)
        set!(ocean.model, T = 20, S = 35)
        time_step!(model, 60)
        state = model.interfaces.exchanger.radiation.state
        @test only(Array(interior(state.ℐꜜˢʷ))) ≈ -250
        @test only(Array(interior(state.ℐꜜˡʷ))) ≈ -350
        @test iteration(model) == 1
    end
end

@testset "Dataset-backed PrescribedRadiation" begin
    for arch in test_architectures
        A = typeof(arch)
        @info "Testing dataset-backed PrescribedRadiation on $A..."

        radiation = synthetic_prescribed_radiation(arch)

        @test radiation isa PrescribedRadiation
        @test radiation.downwelling_shortwave isa FieldTimeSeries
        @test radiation.downwelling_longwave isa FieldTimeSeries
        @test radiation.surface_properties.ocean isa SurfaceRadiationProperties
    end
end

@testset "Radiative cooling when column ice enters an empty cell" begin
    for arch in test_architectures, FT in (Float32, Float64)
        ocean_grid = RectilinearGrid(arch, FT; size = 2, z = (-20, 0), topology = (Flat, Flat, Bounded))
        ice_grid = RectilinearGrid(arch, FT; size = 8, z = SeaIceColumnDiscretization((0, 1)),
                                  topology = (Flat, Flat, Bounded))
        atmosphere_grid = RectilinearGrid(arch, FT; size = 1, z = (0, 10), topology = (Flat, Flat, Bounded))
        ocean = ocean_simulation(ocean_grid; momentum_advection = nothing, tracer_advection = nothing,
                                 closure = nothing, radiative_forcing = nothing)
        set!(ocean.model; T = -1.8, S = 35)
        sea_ice = sea_ice_simulation(ice_grid, ocean; thermodynamics = :column, dynamics = nothing,
                                    advection = nothing, column_bulk_salinity = FixedDrainedIceSalinityProfile(FT))
        set!(sea_ice.model; h = 0, ℵ = 0, hs = 0)
        atmosphere = PrescribedAtmosphere(atmosphere_grid)
        set!(atmosphere; T = 250, p = 101325, q = 0.0005, u = 5, v = 0)
        radiation = PrescribedRadiation(atmosphere_grid)
        set!(radiation; downwelling_shortwave = 0, downwelling_longwave = 100)
        coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
        @test minimum(sea_ice.model.external_heat_fluxes.top.temperature_derivative) > 0

        # Ice arrives after the forcing update.
        set!(sea_ice.model; h = 6, ℵ = 0.1, hs = 0.1)
        thermodynamics = sea_ice.model.ice_thermodynamics
        salinity = Array(interior(thermodynamics.fields.bulk_salinity))
        liquidus = thermodynamics.relation.phase_transitions.liquidus
        temperature = FT(1.1) .* melting_temperature.(Ref(liquidus), salinity)
        set!(thermodynamics; temperature)
        for _ in 1:8
            time_step!(sea_ice.model, 5400)
            @test minimum(sea_ice.model.snow_thermodynamics.top_surface_temperature) > -273.15
            update_state!(coupled_model)
        end
        @test all(isfinite, Array(interior(thermodynamics.fields.internal_energy)))
    end
end
