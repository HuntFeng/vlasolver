#pragma once
#include "../grid.hpp"
#include "../poisson.hpp"
#include <Kokkos_Core.hpp>

/**
 * World struct contains physical properties of the particles, fields, and the immersed boundary.
 */
template <typename WorldType>
struct World {
    Grid& grid;
    Kokkos::View<double*****> f; // distribution function f(x,y,vx,vy) of ion and electron
    // Kokkos::View<double****> flux; // storing fluxes to update distribution function
    Kokkos::View<double****> flux_l; // storing fluxes to update distribution function
    Kokkos::View<double****> flux_r; // storing fluxes to update distribution function
    // Kokkos::View<double****> flux_1st; // storing first order fluxes to update distribution function
    Kokkos::View<double****> flux_1st_l; // storing first order fluxes to update distribution function
    Kokkos::View<double****> flux_1st_r; // storing first order fluxes to update distribution function
    // Kokkos::View<double****> ep; // storing first order fluxes to update distribution function
    Kokkos::View<double****> ep_l;  // storing first order fluxes to update distribution function
    Kokkos::View<double****> ep_r;  // storing first order fluxes to update distribution function
    Kokkos::View<double***> n;      // number density of species
    Kokkos::View<double**> rho;     // ion charge density
    Kokkos::View<double**> phi;     // potential field (assuming Boltzmann distribution for electron)
    Kokkos::View<double***> E;      // Ex(x,y), Ey(x,y)
    Kokkos::View<double***> normal; // n1(x,y), n2(x,y) unit normal vector
    Kokkos::View<double**> eta;     // surface field S(x,y), filled by construct_surface()
    Kokkos::View<double**> eps_p;   // permittivity (eta > 0), filled by construct_permittivity()
    Kokkos::View<double**> eps_m;   // permittivity (eta < 0), filled by construct_permittivity()
    Kokkos::View<double**> jump_a;  // jump condition [phi]_Gamma, filled by poisson_jump_conditions()
    Kokkos::View<double**> jump_b;  // jump condition [d(phi)/dn]_Gamma, filled by poisson_jump_conditions()
    Kokkos::View<PoissonBCPair**> poisson_bc_map;
    Kokkos::Array<double, 2> m = {1.0, 1836.0};     // relative mass of electrons and ions
    Kokkos::Array<double, 2> q = {-1.0, 1.0};       // charge number of electrons and ions
    Kokkos::Array<double, 2> T = {1.0, 1.0 / 10.0}; // relative temperature of electrons and ions

    // simulation time control
    double dt           = 0.0; // time step size
    double total_time   = 1.0; // total simulation time
    size_t total_steps  = 1;   // number of total steps
    size_t diag_steps   = 1;   // number of steps between diagnostics
    size_t current_step = 0;   // current step in the simulation

    World(Grid& grid)
        : grid(grid) {
        // Initialize the views with appropriate dimensions
        auto [nx, ny, nvx, nvy] = grid.ncells;
        f                       = Kokkos::View<double*****>("f", nx, ny, nvx, nvy, 2);
        flux_l                  = Kokkos::View<double****>("flux_l", nx, ny, nvx, nvy);
        flux_r                  = Kokkos::View<double****>("flux_r", nx, ny, nvx, nvy);
        flux_1st_l              = Kokkos::View<double****>("flux_1st_l", nx, ny, nvx, nvy);
        flux_1st_r              = Kokkos::View<double****>("flux_1st_r", nx, ny, nvx, nvy);
        ep_l                    = Kokkos::View<double****>("ep_l", nx, ny, nvx, nvy);
        ep_r                    = Kokkos::View<double****>("ep_r", nx, ny, nvx, nvy);
        n                       = Kokkos::View<double***>("n", nx, ny, 2);
        rho                     = Kokkos::View<double**>("rho", nx, ny);
        phi                     = Kokkos::View<double**>("phi", nx, ny);
        E                       = Kokkos::View<double***>("E", nx, ny, 2);
        normal                  = Kokkos::View<double***>("norm_vec", nx, ny, 2);
        eta                     = Kokkos::View<double**>("eta", nx, ny);
        eps_p                   = Kokkos::View<double**>("eps_p", nx, ny);
        eps_m                   = Kokkos::View<double**>("eps_m", nx, ny);
        jump_a                  = Kokkos::View<double**>("jump_a", nx, ny);
        jump_b                  = Kokkos::View<double**>("jump_b", nx, ny);
        poisson_bc_map          = Kokkos::View<PoissonBCPair**>("poisson_bc_map", nx, ny);
        Kokkos::deep_copy(f, 0.0);
        Kokkos::deep_copy(flux_l, 0.0);
        Kokkos::deep_copy(flux_r, 0.0);
        Kokkos::deep_copy(flux_1st_l, 0.0);
        Kokkos::deep_copy(flux_1st_r, 0.0);
        Kokkos::deep_copy(ep_l, 0.0);
        Kokkos::deep_copy(ep_r, 0.0);
        Kokkos::deep_copy(n, 0.0);
        Kokkos::deep_copy(rho, 0.0);
        Kokkos::deep_copy(phi, 0.0);
        Kokkos::deep_copy(E, 0.0);
        Kokkos::deep_copy(normal, 0.0);
        Kokkos::deep_copy(eta, 0.0);
        Kokkos::deep_copy(eps_p, 1.0);
        Kokkos::deep_copy(eps_m, 1.0);
        Kokkos::deep_copy(jump_a, 0.0);
        Kokkos::deep_copy(jump_b, 0.0);

        // NOTE: construct_normal_field() reads the surface field `eta`, so the derived
        // World must fill `eta` (via construct_surface()) and then call
        // construct_normal_field() from its own constructor.
    }

    KOKKOS_INLINE_FUNCTION
    bool isclose(double val1, double val2, double rtol = 1e-12, double atol = 1e-12) const {
        return Kokkos::abs(val1 - val2) <= atol + rtol * Kokkos::abs(val2);
    }

    void construct_normal_field() {
        auto& grid              = this->grid;
        auto& normal            = this->normal;
        auto& eta               = this->eta;
        auto [nx, ny, nvx, nvy] = grid.ncells;
        auto [dx, dy, dvx, dvy] = grid.spacing(0);
        int ngc                 = grid.ngc;
        // pre-compute normal field from the surface field `eta`
        using Kokkos::abs;
        using Kokkos::pow;
        using Kokkos::sqrt;

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy({ngc, ngc}, {nx - ngc, ny - ngc}), KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                double dx_eta = (-eta(i + 2, j) + 8 * eta(i + 1, j) - 8 * eta(i - 1, j) + eta(i - 2, j)) / (12 * dx);
                double dy_eta = (-eta(i, j + 2) + 8 * eta(i, j + 1) - 8 * eta(i, j - 1) + eta(i, j - 2)) / (12 * dy);
                double norm   = sqrt(pow(dx_eta, 2) + pow(dy_eta, 2));

                // normal field
                if (isclose(norm, 0.0)) {
                    normal(i, j, 0) = 0.0;
                    normal(i, j, 1) = 0.0;
                } else {
                    normal(i, j, 0) = dx_eta / norm;
                    normal(i, j, 1) = dy_eta / norm;
                }
            });
    }

    /**
     * Fill the surface field `eta`.
     * eta(i,j) = S(x,y), the signed value of the immersed boundary expression.
     * S = 0 is the surface of the immersed boundary.
     * S < 0 is the exterior of the computational domain (interior of the immersed object).
     * S > 0 is the interior of the computational domain (exterior of the immersed object).
     * Must be filled over the full domain (including ghost cells).
     * This function will be called by the derived World constructor.
     */
    void construct_surface() { static_cast<WorldType*>(this)->construct_surface(); }

    /**
     * Fill the permittivity field `eps` as a function of spatial coordinate.
     * Must be filled over the full domain (including ghost cells).
     * This function will be called by the derived World constructor.
     */
    void construct_permittivity() { static_cast<WorldType*>(this)->construct_permittivity(); }

    /**
     * Fill the Poisson jump condition fields `jump_a` ([phi]_Gamma) and
     * `jump_b` ([d(phi)/dn]_Gamma) over the full domain.
     * Because this is a host method filling fields, it has full access to all
     * World/derived state, so time-dependent jump conditions are straightforward.
     * This function will be called by the Poisson solver before each solve.
     */
    void poisson_jump_conditions() { static_cast<WorldType*>(this)->poisson_jump_conditions(); }

    /**
     * Initialize the particle distribution function.
     * This function will be called by Vlasov solver
     */
    void initialize_distribution() { static_cast<WorldType*>(this)->initialize_distribution(); };

    /**
     * Boundary conditions for the particle distribution function.
     * This function will be called by Vlasov solver
     */
    void particle_boundary_conditions() { static_cast<WorldType*>(this)->particle_boundary_conditions(); };

    /**
     * Apply boundary conditions to the potential field.
     * This function will be called by Poisson solver
     */
    void potential_boundary_conditions() { static_cast<WorldType*>(this)->potential_boundary_conditions(); };
};
