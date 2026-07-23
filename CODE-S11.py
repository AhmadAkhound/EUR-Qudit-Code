"""
Identifier: Code S11

Purpose
-------
Computes the entropy deficit

    Delta_S(t,d) = ln(d) - S(M(t)) = D(t),

over the sampled (t,d) domain and generates a three-dimensional surface
plot of this quantity.

The code also:

1. Reports the maximum and minimum values on the sampled grid.
2. Detects every local maximum and minimum along t for every integer
   dimension d=2,...,20.
3. Refines each grid-detected extremum by bounded scalar optimization.
4. Prints complete tables of all grid-based and refined extrema.
5. Computes peak-to-trough amplitudes for every dimension.
6. Tests whether these amplitudes increase monotonically with d.
7. Tests whether the entropy deficit is positive at every sampled point.
8. Saves the sampled surface, all extrema, and the dimension-dependent
   amplitude table as CSV files.

Interpretation
--------------
The plotted quantity is exactly equal to the closed-form discord D(t).
It is not interpreted as an EPR-steering violation strength. This code
does not construct an assemblage, test a local-hidden-state model, or
evaluate an entropic steering inequality.

Numerical scope
---------------
The physical relations defining Gamma(t), r(t), M(t), and D(t) are
analytical, but their values are evaluated numerically using quadrature
and Hermitian eigensolvers.

The reported grid extrema refer only to

    d = 2,...,20,
    0.01 <= t <= 15,
    400 sampled time points.

The refined extrema are local numerical optimizations initialized from
the grid-detected extrema. They are not claimed to be mathematically
exact global extrema over a continuous or unbounded parameter domain.

Used in
-------
Figure 6: entropy-deficit / discord landscape.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize
from scipy.signal import argrelextrema


# ============================================================
# Physical model
# ============================================================

def Gamma(t, omega0=1.0, lam=0.1, eta=0.15):
    """
    Numerically evaluate the pure-dephasing exponent Gamma(t).

    The integration interval and spectral density are the same as in the
    other supplementary codes.
    """
    def integrand(w):
        J = eta * lam**2 / ((w - omega0)**2 + lam**2)
        return J / w**2 * (1.0 - np.cos(w * t))

    val, err = integrate.quad(
        integrand,
        1.0e-6,
        50.0 * omega0,
        limit=400,
        epsabs=1.0e-14,
        epsrel=1.0e-14,
    )

    if not np.isfinite(val):
        raise RuntimeError(
            f"Non-finite quadrature result at t={t}: value={val}, error={err}"
        )

    # A warning rather than a fatal error is preferable here because QUADPACK
    # can report a conservative error estimate near machine precision.
    tolerance = 1.0e-9 * max(1.0, abs(val))
    if err > tolerance:
        print(
            "WARNING: comparatively large quadrature error estimate: "
            f"t={t:.10f}, value={val:.12e}, error={err:.3e}"
        )

    return val


def r_of_t(t, omega0=1.0, lam=0.1, eta=0.15):
    """Return the decoherence factor r(t)=exp[-2 Gamma(t)]."""
    gamma_value = Gamma(t, omega0=omega0, lam=lam, eta=eta)
    return np.exp(-2.0 * gamma_value)


def toeplitz_density_matrix(r, d):
    """
    Construct the d x d maximally correlated coefficient matrix

        M_nm = r^((n-m)^2) / d.
    """
    indices = np.arange(d)
    differences = indices[:, None] - indices[None, :]
    matrix = r ** (differences**2) / d

    # Explicit symmetrization suppresses negligible floating-point asymmetry.
    matrix = 0.5 * (matrix + matrix.T)
    return matrix


def von_neumann_entropy(rho, eigenvalue_cutoff=1.0e-14):
    """
    Compute S(rho)=-Tr[rho ln(rho)] from the Hermitian eigenvalues.

    Tiny negative eigenvalues attributable to roundoff are clipped only
    after a positivity check.
    """
    eigs = np.linalg.eigvalsh(rho)
    eigs = np.real(eigs)

    minimum_eigenvalue = np.min(eigs)
    if minimum_eigenvalue < -1.0e-11:
        raise RuntimeError(
            "Matrix is not positive semidefinite within numerical accuracy: "
            f"minimum eigenvalue={minimum_eigenvalue:.6e}"
        )

    eigs = np.clip(eigs, 0.0, None)

    trace = np.sum(eigs)
    if not np.isclose(trace, 1.0, rtol=1.0e-10, atol=1.0e-12):
        raise RuntimeError(
            f"Density-matrix trace differs from one: trace={trace:.16e}"
        )

    positive = eigs[eigs > eigenvalue_cutoff]
    return -np.sum(positive * np.log(positive))


def entropy_toeplitz(r, d):
    """Return S(M(t)) for a specified r and d."""
    matrix = toeplitz_density_matrix(r, d)
    return von_neumann_entropy(matrix)


def entropy_deficit(t, d, omega0=1.0, lam=0.1, eta=0.15):
    """
    Return

        Delta_S(t,d) = ln(d)-S(M(t)),

    which equals the closed-form discord D(t) for the state family studied
    in the manuscript.
    """
    r = r_of_t(t, omega0=omega0, lam=lam, eta=eta)
    entropy = entropy_toeplitz(r, d)
    deficit = np.log(d) - entropy

    # Suppress a possible negative value at the level of roundoff only.
    if deficit < 0.0 and abs(deficit) < 1.0e-12:
        deficit = 0.0

    return deficit


# ============================================================
# Extremum detection and refinement
# ============================================================

def refine_local_extremum(
    d,
    t_grid,
    values,
    grid_index,
    extremum_type,
    omega0=1.0,
    lam=0.1,
    eta=0.15,
):
    """
    Refine one grid-detected local extremum inside the neighboring grid
    interval using bounded scalar minimization.

    For a maximum, minimize -Delta_S.
    For a minimum, minimize +Delta_S.
    """
    if grid_index <= 0 or grid_index >= len(t_grid) - 1:
        raise ValueError("A local extremum must have two neighboring points.")

    left = float(t_grid[grid_index - 1])
    right = float(t_grid[grid_index + 1])

    if extremum_type == "maximum":
        objective = lambda t: -entropy_deficit(
            t,
            d,
            omega0=omega0,
            lam=lam,
            eta=eta,
        )
    elif extremum_type == "minimum":
        objective = lambda t: entropy_deficit(
            t,
            d,
            omega0=omega0,
            lam=lam,
            eta=eta,
        )
    else:
        raise ValueError(
            "extremum_type must be either 'maximum' or 'minimum'."
        )

    result = optimize.minimize_scalar(
        objective,
        bounds=(left, right),
        method="bounded",
        options={
            "xatol": 1.0e-11,
            "maxiter": 500,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"Extremum refinement failed for d={d}, "
            f"type={extremum_type}, grid index={grid_index}: "
            f"{result.message}"
        )

    refined_t = float(result.x)
    refined_value = float(
        entropy_deficit(
            refined_t,
            d,
            omega0=omega0,
            lam=lam,
            eta=eta,
        )
    )

    return {
        "dimension": int(d),
        "type": extremum_type,
        "grid_index": int(grid_index),
        "grid_t": float(t_grid[grid_index]),
        "grid_value": float(values[grid_index]),
        "left_bound": left,
        "right_bound": right,
        "refined_t": refined_t,
        "refined_value": refined_value,
        "optimizer_success": bool(result.success),
        "optimizer_nfev": int(result.nfev),
    }


def find_all_local_extrema(
    d,
    t_grid,
    values,
    omega0=1.0,
    lam=0.1,
    eta=0.15,
):
    """
    Detect and refine all strict local maxima and minima along the sampled
    time profile for one dimension.
    """
    maxima_indices = argrelextrema(values, np.greater, order=1)[0]
    minima_indices = argrelextrema(values, np.less, order=1)[0]

    extrema = []

    for idx in maxima_indices:
        extrema.append(
            refine_local_extremum(
                d=d,
                t_grid=t_grid,
                values=values,
                grid_index=int(idx),
                extremum_type="maximum",
                omega0=omega0,
                lam=lam,
                eta=eta,
            )
        )

    for idx in minima_indices:
        extrema.append(
            refine_local_extremum(
                d=d,
                t_grid=t_grid,
                values=values,
                grid_index=int(idx),
                extremum_type="minimum",
                omega0=omega0,
                lam=lam,
                eta=eta,
            )
        )

    # Sort all extrema chronologically.
    extrema.sort(key=lambda item: item["refined_t"])
    return extrema


# ============================================================
# Output helpers
# ============================================================

def print_separator(width=118, character="="):
    print(character * width)


def print_extrema_table(extrema):
    """Print one complete table containing every detected extremum."""
    print_separator()
    print("COMPLETE TABLE OF ALL LOCAL EXTREMA FOR d=2,...,20")
    print_separator()

    header = (
        f"{'d':>3}  "
        f"{'type':>8}  "
        f"{'grid index':>10}  "
        f"{'grid t':>13}  "
        f"{'grid value':>16}  "
        f"{'refined t':>15}  "
        f"{'refined value':>18}  "
        f"{'interval':>25}  "
        f"{'nfev':>6}"
    )
    print(header)
    print("-" * len(header))

    for item in extrema:
        interval_text = (
            f"[{item['left_bound']:.8f},"
            f" {item['right_bound']:.8f}]"
        )

        print(
            f"{item['dimension']:3d}  "
            f"{item['type']:>8}  "
            f"{item['grid_index']:10d}  "
            f"{item['grid_t']:13.8f}  "
            f"{item['grid_value']:16.10f}  "
            f"{item['refined_t']:15.10f}  "
            f"{item['refined_value']:18.12f}  "
            f"{interval_text:>25}  "
            f"{item['optimizer_nfev']:6d}"
        )

    print_separator()


def print_extrema_grouped_by_dimension(extrema, dimensions):
    """Print a compact chronological list separately for every dimension."""
    print("\n")
    print_separator()
    print("LOCAL MAXIMA AND MINIMA GROUPED BY DIMENSION")
    print_separator()

    for d in dimensions:
        entries = [
            item for item in extrema if item["dimension"] == int(d)
        ]

        print(f"\n--- d={d} ---")

        if not entries:
            print("No strict interior local extrema detected.")
            continue

        for item in entries:
            label = "PEAK  " if item["type"] == "maximum" else "TROUGH"
            print(
                f"{label}: "
                f"t_grid={item['grid_t']:.8f}, "
                f"value_grid={item['grid_value']:.12f}, "
                f"t_refined={item['refined_t']:.10f}, "
                f"value_refined={item['refined_value']:.12f}"
            )


def save_surface_csv(path, dimensions, times, surface):
    """
    Save the entire sampled surface.

    The first column is d; all remaining columns correspond to sampled t.
    """
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        header = ["d"] + [f"t={t:.12f}" for t in times]
        writer.writerow(header)

        for i, d in enumerate(dimensions):
            writer.writerow(
                [int(d)] + [f"{value:.16e}" for value in surface[i]]
            )


def save_extrema_csv(path, extrema):
    """Save every grid-based and refined local extremum."""
    fieldnames = [
        "dimension",
        "type",
        "grid_index",
        "grid_t",
        "grid_value",
        "left_bound",
        "right_bound",
        "refined_t",
        "refined_value",
        "optimizer_success",
        "optimizer_nfev",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(extrema)


def save_amplitude_csv(path, amplitude_rows):
    """Save peak/trough amplitudes for every dimension."""
    fieldnames = [
        "dimension",
        "first_peak_t",
        "first_peak_value",
        "first_trough_t",
        "first_trough_value",
        "first_peak_minus_first_trough",
        "largest_local_maximum",
        "smallest_local_minimum",
        "full_local_range",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(amplitude_rows)


# ============================================================
# Main calculation
# ============================================================

def main():
    # Environmental parameters used for Figure 6.
    omega0 = 1.0
    lam = 0.1
    eta = 0.15

    # Sampled domain.
    dimensions = np.arange(2, 21, dtype=int)
    times = np.linspace(0.01, 15.0, 400)

    output_directory = Path("code_s11_results")
    output_directory.mkdir(parents=True, exist_ok=True)

    print("Computing the entropy-deficit surface...")
    print(
        f"Parameters: omega0={omega0}, lambda={lam}, eta={eta}"
    )
    print(
        f"Sampled domain: d={dimensions[0]},...,{dimensions[-1]}, "
        f"t in [{times[0]}, {times[-1]}], "
        f"{len(times)} time points"
    )

    # --------------------------------------------------------
    # Compute the full sampled surface.
    # --------------------------------------------------------

    surface = np.zeros((len(dimensions), len(times)), dtype=float)

    for i, d in enumerate(dimensions):
        for j, t in enumerate(times):
            surface[i, j] = entropy_deficit(
                t,
                int(d),
                omega0=omega0,
                lam=lam,
                eta=eta,
            )

        print(f"Completed d={d}")

    # --------------------------------------------------------
    # Grid extrema over the sampled domain.
    # --------------------------------------------------------

    maximum_indices = np.unravel_index(
        np.argmax(surface),
        surface.shape,
    )
    minimum_indices = np.unravel_index(
        np.argmin(surface),
        surface.shape,
    )

    i_max, j_max = maximum_indices
    i_min, j_min = minimum_indices

    grid_maximum = float(surface[i_max, j_max])
    grid_minimum = float(surface[i_min, j_min])

    print("\n")
    print_separator()
    print("GRID EXTREMA OVER THE SAMPLED (t,d) DOMAIN")
    print_separator()

    print(
        f"Grid maximum: value={grid_maximum:.12f}, "
        f"d={dimensions[i_max]}, "
        f"t={times[j_max]:.12f}, "
        f"grid indices=({i_max},{j_max})"
    )

    print(
        f"Grid minimum: value={grid_minimum:.12f}, "
        f"d={dimensions[i_min]}, "
        f"t={times[j_min]:.12f}, "
        f"grid indices=({i_min},{j_min})"
    )

    # Boundary profiles are printed because the largest/smallest sampled
    # values may occur at an endpoint and are therefore not interior local
    # extrema.
    print("\nBoundary values for every dimension:")
    print(
        f"{'d':>3}  "
        f"{'Delta(t_min)':>18}  "
        f"{'Delta(t_max)':>18}"
    )
    print("-" * 45)

    for i, d in enumerate(dimensions):
        print(
            f"{d:3d}  "
            f"{surface[i, 0]:18.12f}  "
            f"{surface[i, -1]:18.12f}"
        )

    # --------------------------------------------------------
    # Find and refine all local extrema for all dimensions.
    # --------------------------------------------------------

    all_extrema = []

    for i, d in enumerate(dimensions):
        extrema_d = find_all_local_extrema(
            d=int(d),
            t_grid=times,
            values=surface[i],
            omega0=omega0,
            lam=lam,
            eta=eta,
        )
        all_extrema.extend(extrema_d)

    print_extrema_table(all_extrema)
    print_extrema_grouped_by_dimension(all_extrema, dimensions)

    # --------------------------------------------------------
    # Peak-to-trough analysis for every dimension.
    # --------------------------------------------------------

    amplitude_rows = []

    print("\n")
    print_separator()
    print("PEAK-TO-TROUGH ANALYSIS FOR EVERY DIMENSION")
    print_separator()

    header = (
        f"{'d':>3}  "
        f"{'first peak t':>13}  "
        f"{'first peak':>14}  "
        f"{'first trough t':>15}  "
        f"{'first trough':>14}  "
        f"{'peak-trough':>15}  "
        f"{'full local range':>18}"
    )
    print(header)
    print("-" * len(header))

    first_cycle_amplitudes = []
    full_local_ranges = []

    for d in dimensions:
        entries = [
            item for item in all_extrema
            if item["dimension"] == int(d)
        ]

        maxima = [
            item for item in entries if item["type"] == "maximum"
        ]
        minima = [
            item for item in entries if item["type"] == "minimum"
        ]

        maxima.sort(key=lambda item: item["refined_t"])
        minima.sort(key=lambda item: item["refined_t"])

        if not maxima or not minima:
            print(
                f"{d:3d}  insufficient interior extrema for amplitude analysis"
            )
            continue

        # In this model the first trough occurs before the first peak.
        # We report the first revived peak minus the preceding first trough.
        first_peak = maxima[0]
        preceding_minima = [
            item for item in minima
            if item["refined_t"] < first_peak["refined_t"]
        ]

        if preceding_minima:
            first_trough = preceding_minima[-1]
        else:
            first_trough = minima[0]

        first_amplitude = (
            first_peak["refined_value"]
            - first_trough["refined_value"]
        )

        largest_local_maximum = max(
            item["refined_value"] for item in maxima
        )
        smallest_local_minimum = min(
            item["refined_value"] for item in minima
        )
        full_local_range = (
            largest_local_maximum - smallest_local_minimum
        )

        first_cycle_amplitudes.append(
            (int(d), float(first_amplitude))
        )
        full_local_ranges.append(
            (int(d), float(full_local_range))
        )

        amplitude_rows.append(
            {
                "dimension": int(d),
                "first_peak_t": first_peak["refined_t"],
                "first_peak_value": first_peak["refined_value"],
                "first_trough_t": first_trough["refined_t"],
                "first_trough_value": first_trough["refined_value"],
                "first_peak_minus_first_trough": first_amplitude,
                "largest_local_maximum": largest_local_maximum,
                "smallest_local_minimum": smallest_local_minimum,
                "full_local_range": full_local_range,
            }
        )

        print(
            f"{d:3d}  "
            f"{first_peak['refined_t']:13.8f}  "
            f"{first_peak['refined_value']:14.10f}  "
            f"{first_trough['refined_t']:15.8f}  "
            f"{first_trough['refined_value']:14.10f}  "
            f"{first_amplitude:15.10f}  "
            f"{full_local_range:18.10f}"
        )

    # --------------------------------------------------------
    # Explicit monotonicity tests.
    # --------------------------------------------------------

    first_amplitude_values = np.array(
        [value for _, value in first_cycle_amplitudes],
        dtype=float,
    )
    full_range_values = np.array(
        [value for _, value in full_local_ranges],
        dtype=float,
    )

    first_amplitude_differences = np.diff(first_amplitude_values)
    full_range_differences = np.diff(full_range_values)

    first_amplitudes_strictly_increasing = bool(
        np.all(first_amplitude_differences > 0.0)
    )
    full_ranges_strictly_increasing = bool(
        np.all(full_range_differences > 0.0)
    )

    print("\n")
    print_separator()
    print("MONOTONICITY TESTS ACROSS d=2,...,20")
    print_separator()

    print(
        "First revived-peak minus preceding-trough amplitude is "
        f"strictly increasing with d: "
        f"{first_amplitudes_strictly_increasing}"
    )

    print(
        "Largest-local-maximum minus smallest-local-minimum range is "
        f"strictly increasing with d: "
        f"{full_ranges_strictly_increasing}"
    )

    print("\nSuccessive differences in the first-cycle amplitude:")
    for index, difference in enumerate(first_amplitude_differences):
        d_left = first_cycle_amplitudes[index][0]
        d_right = first_cycle_amplitudes[index + 1][0]
        print(
            f"d={d_left} -> d={d_right}: "
            f"Delta amplitude={difference:.12e}"
        )

    print("\nSuccessive differences in the full local range:")
    for index, difference in enumerate(full_range_differences):
        d_left = full_local_ranges[index][0]
        d_right = full_local_ranges[index + 1][0]
        print(
            f"d={d_left} -> d={d_right}: "
            f"Delta range={difference:.12e}"
        )

    # --------------------------------------------------------
    # Positivity test over the sampled grid.
    # --------------------------------------------------------

    numerical_tolerance = 1.0e-12
    minimum_sampled_deficit = float(np.min(surface))
    all_sampled_values_positive = bool(
        np.all(surface > numerical_tolerance)
    )

    print("\n")
    print_separator()
    print("POSITIVITY CHECK ON THE SAMPLED GRID")
    print_separator()

    print(
        f"Minimum sampled entropy deficit: "
        f"{minimum_sampled_deficit:.16e}"
    )
    print(
        f"All sampled values exceed {numerical_tolerance:.1e}: "
        f"{all_sampled_values_positive}"
    )

    # --------------------------------------------------------
    # Save all numerical tables.
    # --------------------------------------------------------

    surface_csv = output_directory / "code_s11_surface_full_grid.csv"
    extrema_csv = output_directory / "code_s11_all_local_extrema.csv"
    amplitude_csv = output_directory / "code_s11_amplitudes_by_dimension.csv"

    save_surface_csv(
        path=surface_csv,
        dimensions=dimensions,
        times=times,
        surface=surface,
    )
    save_extrema_csv(
        path=extrema_csv,
        extrema=all_extrema,
    )
    save_amplitude_csv(
        path=amplitude_csv,
        amplitude_rows=amplitude_rows,
    )

    # --------------------------------------------------------
    # Three-dimensional entropy-deficit surface.
    # --------------------------------------------------------

    time_mesh, dimension_mesh = np.meshgrid(times, dimensions)

    figure = plt.figure(figsize=(10.0, 6.8))
    axis = figure.add_subplot(111, projection="3d")

    surface_plot = axis.plot_surface(
        time_mesh,
        dimension_mesh,
        surface,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=2,
    )

    axis.set_xlabel(r"$t$", labelpad=10)
    axis.set_ylabel(r"$d$", labelpad=10)
    axis.set_zlabel(
        r"$\ln d-S(M(t))=D(t)$",
        labelpad=10,
    )
    axis.set_title(
        r"Entropy deficit $\ln d-S(M(t))=D(t)$ over the sampled $(t,d)$ domain"
    )
    axis.view_init(elev=28, azim=-60)

    colorbar = figure.colorbar(
        surface_plot,
        ax=axis,
        shrink=0.64,
        pad=0.10,
    )
    colorbar.set_label(r"$D(t)$")

    figure.tight_layout()

    pdf_path = output_directory / "figure6_entropy_deficit_surface.pdf"
    png_path = output_directory / "figure6_entropy_deficit_surface.png"

    figure.savefig(pdf_path, dpi=300, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()

    # --------------------------------------------------------
    # Additional two-dimensional profiles for visual inspection.
    # --------------------------------------------------------

    representative_dimensions = [2, 5, 10, 15, 20]

    profile_figure, profile_axis = plt.subplots(figsize=(9.0, 5.8))

    for d in representative_dimensions:
        row_index = int(np.where(dimensions == d)[0][0])
        profile_axis.plot(
            times,
            surface[row_index],
            linewidth=1.8,
            label=rf"$d={d}$",
        )

        extrema_d = [
            item for item in all_extrema
            if item["dimension"] == d
        ]

        maxima_d = [
            item for item in extrema_d
            if item["type"] == "maximum"
        ]
        minima_d = [
            item for item in extrema_d
            if item["type"] == "minimum"
        ]

        if maxima_d:
            profile_axis.scatter(
                [item["refined_t"] for item in maxima_d],
                [item["refined_value"] for item in maxima_d],
                marker="^",
                s=45,
            )

        if minima_d:
            profile_axis.scatter(
                [item["refined_t"] for item in minima_d],
                [item["refined_value"] for item in minima_d],
                marker="v",
                s=45,
            )

    profile_axis.set_xlabel(r"$t$")
    profile_axis.set_ylabel(r"$\ln d-S(M(t))=D(t)$")
    profile_axis.set_title(
        "Entropy-deficit profiles and refined local extrema"
    )
    profile_axis.grid(alpha=0.25)
    profile_axis.legend()
    profile_figure.tight_layout()

    profile_pdf_path = (
        output_directory / "figure6_entropy_deficit_profiles.pdf"
    )
    profile_png_path = (
        output_directory / "figure6_entropy_deficit_profiles.png"
    )

    profile_figure.savefig(
        profile_pdf_path,
        dpi=300,
        bbox_inches="tight",
    )
    profile_figure.savefig(
        profile_png_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # --------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------

    print("\n")
    print_separator()
    print("FILES SAVED")
    print_separator()

    print(f"Full sampled surface:       {surface_csv}")
    print(f"All refined extrema:        {extrema_csv}")
    print(f"Amplitude table:            {amplitude_csv}")
    print(f"Three-dimensional PDF:      {pdf_path}")
    print(f"Three-dimensional PNG:      {png_path}")
    print(f"Time-profile PDF:           {profile_pdf_path}")
    print(f"Time-profile PNG:           {profile_png_path}")

    print("\nCode S11 completed successfully.")


if __name__ == "__main__":
    main()