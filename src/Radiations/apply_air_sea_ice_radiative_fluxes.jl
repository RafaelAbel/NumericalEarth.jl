using Oceananigans.Fields: OneField
using ClimaSeaIce: SeaIceModel
using ClimaSeaIce.SeaIceThermodynamics: LinearizedSurfaceFlux, ColumnEnergyThermodynamics

"""
    apply_air_sea_ice_radiative_fluxes!(coupled_model)

Add the radiative contribution to the net sea-ice top heat flux and write
the diagnostic radiative fluxes (upwelling LW, absorbed LW, transmitted SW)
into `coupled_model.radiation.interface_fluxes.sea_ice`.

When `coupled_model.radiation === nothing`, this is a no-op.
Also a no-op when sea-ice is not a `Simulation{<:SeaIceModel}`.
"""
EarthSystemModels.apply_air_sea_ice_radiative_fluxes!(::EarthSystemModel{<:Nothing}) = nothing

EarthSystemModels.apply_air_sea_ice_radiative_fluxes!(coupled_model::EarthSystemModel) =
    _apply_air_sea_ice_radiative_fluxes_dispatch!(coupled_model, coupled_model.sea_ice)

# No sea-ice or non-prognostic sea-ice: nothing to do.
_apply_air_sea_ice_radiative_fluxes_dispatch!(coupled_model, ::Any) = nothing

function _apply_air_sea_ice_radiative_fluxes_dispatch!(coupled_model::EarthSystemModel,
                                                       sea_ice::Simulation{<:SeaIceModel})
    radiation = coupled_model.radiation
    interface_fluxes = radiation.interface_fluxes
    isnothing(interface_fluxes) && return nothing
    haskey(interface_fluxes, :sea_ice) || return nothing

    grid = sea_ice.model.grid
    arch = architecture(grid)
    clock = coupled_model.clock

    top_heat_flux = coupled_model.interfaces.net_fluxes.sea_ice.top.heat
    radiation_state = coupled_model.interfaces.exchanger.radiation.state
    rk = kernel_radiation_properties(radiation)

    surface_temperature = coupled_model.interfaces.atmosphere_sea_ice_interface.temperature
    sea_ice_properties = coupled_model.interfaces.sea_ice_properties
    top_flux = sea_ice.model.external_heat_fluxes.top
    area_weight = sea_ice_area_weight(sea_ice.model.ice_thermodynamics, sea_ice, eltype(grid))

    launch!(arch, grid, :xy,
            _apply_air_sea_ice_radiative_fluxes!,
            top_heat_flux,
            interface_fluxes.sea_ice,
            grid,
            clock,
            rk,
            radiation_state,
            area_weight,
            surface_temperature,
            sea_ice_properties)

    add_sea_ice_radiative_flux_derivative!(top_flux, arch, grid,
                                           interface_fluxes.sea_ice.upwelling_longwave,
                                           area_weight,
                                           grid,
                                           sea_ice_properties)

    return nothing
end

# Column ice carries fluxes per unit ice area. Other ice carries them per unit cell area.
sea_ice_area_weight(thermodynamics, sea_ice, FT) = sea_ice_concentration(sea_ice)
sea_ice_area_weight(::ColumnEnergyThermodynamics, sea_ice, FT) = OneField(FT)

# Only a linearized top flux needs a temperature derivative.
add_sea_ice_radiative_flux_derivative!(top_flux, args...) = nothing

function add_sea_ice_radiative_flux_derivative!(top_flux::LinearizedSurfaceFlux, arch, grid, args...)
    launch!(arch, grid, :xy,
            _add_sea_ice_radiative_flux_derivative!,
            top_flux.temperature_derivative,
            top_flux.reference_temperature,
            args...)
    return nothing
end

@kernel function _apply_air_sea_ice_radiative_fluxes!(top_heat_flux,
                                                      interface_radiative_flux,
                                                      grid,
                                                      clock,
                                                      rk,
                                                      radiation_state,
                                                      area_weight,
                                                      surface_temperature,
                                                      sea_ice_properties)

    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)
    time = Time(clock.time)

    @inbounds begin
        weight = area_weight[i, j, 1]
        Ts = surface_temperature[i, j, kᴺ]
    end
    Ts = convert_to_kelvin(sea_ice_properties.temperature_units, Ts)

    rs = air_sea_ice_interface_radiation_state(rk, radiation_state, i, j, kᴺ, grid, time)

    ℐꜛˡʷ = emitted_longwave_radiation(Ts, rs.σ, rs.ϵ)
    ℐₐˡʷ = absorbed_longwave_radiation(rs.ϵ, rs.ℐꜜˡʷ)
    ℐₜˢʷ = transmitted_shortwave_radiation(rs.α, rs.ℐꜜˢʷ)
    ΣQ  = (ℐꜛˡʷ + ℐₐˡʷ + ℐₜˢʷ) * weight

    inactive = inactive_node(i, j, kᴺ, grid, Center(), Center(), Center())

    @inbounds begin
        top_heat_flux[i, j, 1] += ifelse(inactive, zero(grid), ΣQ)
        interface_radiative_flux.upwelling_longwave[i, j, 1]    = ℐꜛˡʷ
        interface_radiative_flux.downwelling_longwave[i, j, 1]  = ℐₐˡʷ
        interface_radiative_flux.downwelling_shortwave[i, j, 1] = ℐₜˢʷ
    end
end

@kernel function _add_sea_ice_radiative_flux_derivative!(temperature_derivative,
                                                          reference_temperature,
                                                          upwelling_longwave,
                                                          area_weight,
                                                          grid,
                                                          sea_ice_properties)
    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)
    @inbounds begin
        weight = area_weight[i, j, 1]
        Tref = reference_temperature[i, j, 1]
        Qlw = upwelling_longwave[i, j, 1]
    end
    Tk = convert_to_kelvin(sea_ice_properties.temperature_units, Tref)
    inactive = inactive_node(i, j, kᴺ, grid, Center(), Center(), Center())
    slope = ifelse(inactive, zero(eltype(grid)), 4 * Qlw / Tk * weight)
    @inbounds temperature_derivative[i, j, 1] += slope
end
