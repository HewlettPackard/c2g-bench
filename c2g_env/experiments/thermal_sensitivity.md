# Thermal Sensitivity Experiment

## Purpose

This experiment tests whether the benchmark's thermal-safety results depend on
permissive aggregate thermal assumptions. It complements, rather than
replaces, facility calibration, driving the implemented `ThermalTwin` under
one-at-a-time parameter variations, named coupled cases, and three frozen
analytic controllers: bang-bang, PID, and rule-based.

The swept capacitances and conductances are facility-level effective
parameters, not intrinsic material properties. Material and equipment
specifications inform their construction, but inventories, geometry, flow,
equipment count, and thermal participation fix the aggregate values.

The safety anchors remain fixed throughout:

| Quantity | Fixed value | Role |
| --- | ---: | --- |
| Warning temperature | 33 degC | Thermal-warning violation threshold |
| Safety temperature | 35 degC | `C2GFastEnv` thermal termination threshold |
| Simulation step | 5 s | Benchmark low-level control interval |
| Facility IT nameplate | 250 MW | 150 MW Zone A plus 100 MW Zone B |

## Sweep

The discrete grids are stored in `conf/experiments.yaml`; the aggregate bounds
are physically interpreted facility cases, not material-property tolerances.

Status labels: **N**, benchmark nominal point; **O**, ordinary or physically
plausible operating/design case; **B**, boundary or conditional case; **NR**,
not recommended for continuous full-nameplate operation because the
maximum-cooling test crossed 35 degC; **S**, declared stress or failure
condition. A value carries two labels when a standard permits it but the
current plant sizing cannot sustain it.

The **NR** values fail the open-loop stress test, in which IT load is pinned
at the worst-case maximum and cooling at full effort, yet the zone still
crosses 35 degC. The pure-NR values would not be selected as a sizing basis
for this 150/100 MW plant; the S/NR values additionally represent runtime
degradation of an otherwise adequate plant rather than a design choice. We
retain all of them in the sweep as boundary conditions rather than as
recommended operating points.

### Label Basis

Each label combines two kinds of evidence rather than a single test:

- **A priori, specification plausibility (N, O, B, S).** Each non-nominal
  value is checked against a physical reference (ASHRAE 90.1/127/TC 9.9,
  NIST/IAPWS material data, IECC C402, AHRI 550/590 or 551/591, or vendor
  specifications) and classified as ordinary and plausible (**O**), boundary
  or conditional if it needs a further physical assumption to justify (**B**),
  or a declared stress or failure case such as an equipment outage or extreme
  weather (**S**). **N** marks the current benchmark default.
- **A posteriori, the controller-free maximum-cooling probe (NR).** A value is
  labelled **NR** when, run at constant worst-case load (150/100 MW) with
  cooling pinned to maximum effort (`open_maxcool`: `pump_speed=1.0`,
  `hvac_effort=1.0`), it still crosses the fixed 35 degC line. This is the one
  label tied to a simulation result rather than a specification. NR is decided
  from `open_maxcool` alone, the most favourable cooling policy: minimum
  cooling (0.15, 0.0) crosses in all 48 configurations and fixed 0.7 cooling in
  45 of 48, so neither would isolate under-provisioning. NR therefore flags
  configurations that stay infeasible even under the best available cooling.
- **Dual labels** (`N/B`, `B/NR`, `S/NR`) appear when the two kinds of
  evidence disagree: a specification permits the value, but the maximum-cooling
  probe shows the current plant sizing cannot sustain it at nameplate.

These labels are written by hand in the tables below, not computed. The
`stress_condition` column in the Full-Environment Cross-Scenario Evaluation
re-derives a coarser, NR-only version of this taxonomy for per-episode fault
analysis.

| Parameter | Annotated grid | Physical interpretation | Constituent materials or specification basis |
| --- | --- | --- | --- |
| $C_A$ (MJ/K) | 13,500 **[B: low inertia]**; 20,250 **[O]**; 27,000 **[N]**; 40,500 **[O: high inertia]** | Effective Zone A inventory; about 6.3, 9.5, 12.7, and 19.0 min at nominal full-cooling conductance. Capacitance does not determine steady-state feasibility, so the low value is conservative rather than intrinsically unsafe. | $C=\sum_i \pi_i m_i c_{p,i}$. Water: $c_p\approx4.18$ kJ/kg/K (NIST/IAPWS). A 30% propylene-glycol mixture: about 3.9 kJ/kg/K (ASHRAE/Dow). Iron/steel proxy: about 0.45, aluminum: about 0.90, copper: about 0.385 kJ/kg/K (NIST-JANAF). Concrete: about 0.84 to 0.88 kJ/kg/K (ASHRAE/ISO 10456). Aggregate values require assumed inventories and participation fractions $\pi_i$. |
| $C_B$ (MJ/K) | 5,000 **[B: low inertia]**; 7,500 **[O]**; 10,000 **[N]**; 15,000 **[O: high inertia]** | Effective Zone B inventory; about 6.2, 9.3, 12.3, and 18.5 min at nominal full-HVAC conductance. | Dry air contributes only about 1.19 kJ/m3/K (NIST/ASHRAE). The aggregate therefore also requires rack metals, coil/chilled water, and participating structural mass. Air alone cannot support the selected values. |
| $K_{\mathrm{liq}}$ (MW/K) | 18.8 **[NR]**; 25.1 **[NR]**; 35.0 **[N]**; 50.1 **[O: high capacity]** | 150 MW removal at effective approaches of about 8, 6, 4.3, and 3 K. The first two values are plausible under-sized or degraded configurations, but cannot maintain the modeled 35 degC limit at continuous 150 MW. | Cray EX QuickSpecs specify a 1.6 MW in-row CDU and a 70 kW in-rack CDU. CoolIT CHx2000 and CHx80 provide 2 MW and 80 kW equipment references. Convert rated heat removal using $K=Q/\Delta T_{\mathrm{lm}}$. The approach temperatures remain declared design assumptions until supported by vendor submittals. |
| $K_{\mathrm{air}}$ (MW/K) | 6.7 **[B: low headroom]**; 10.0 **[O]**; 13.0 **[N]**; 20.0 **[O: high capacity]** | 100 MW removal at effective zone-to-supply differences of about 15, 10, 7.7, and 5 K. The 6.7 case remains below 35 degC only at maximum HVAC effort under nameplate load. | Vertiv Liebert CW units span roughly 33 to 517 kW and about 6,050 to 60,000 CFM. Aggregate $K$ requires model count, active airflow, rated temperatures, and a declared mapping between the benchmark state $T_B$ and measured equipment inlet or return air. |
| $K_{\mathrm{env},A}$ (MW/K) | 0.05 **[O: passive]**; 0.10 **[O: passive]**; 0.25 **[B: ventilation-inclusive]**; 0.50 **[N/B: ventilation-inclusive]** | Code envelope plus infiltration through ventilation-inclusive ambient coupling. The current 0.50 value is not recommended as a passive-fabric interpretation. | ASHRAE 90.1 and IECC C402 give roof $U\approx0.12$ to 0.22 W/m2/K, wall $U\approx0.21$ to 0.59 W/m2/K, and fixed fenestration $U\approx1.6$ to 2.8 W/m2/K. Compute $K_{\mathrm{env}}=\sum_i U_iA_i+FP+\rho c_p\dot V$. Values 0.25 to 0.50 MW/K require substantial ventilation/economizer coupling for the assumed campus geometry. |
| $K_{\mathrm{env},B}$ (MW/K) | 0.05 **[O: passive]**; 0.10 **[O: passive]**; 0.25 **[B: ventilation-inclusive]**; 0.50 **[N/B: ventilation-inclusive]** | Same interpretation with separate zone geometry. | Same ASHRAE 90.1, IECC C402, and DOE-PNNL prototype-building basis. The nominal 0.5 MW/K should be treated as ventilation-inclusive, not passive fabric conductance. |
| $\mathrm{COP}_{\mathrm{air,base}}$ | 2.5 **[B: low efficiency]**; 3.5 **[N]**; 5.0 **[O: high efficiency]** | Low-, nominal-, and high-efficiency air-plant cases. The 2.5 case is not itself beyond the thermal limit at maximum HVAC effort, but is a poor-efficiency boundary and becomes capacity-limited at 0.7 effort. | ASHRAE 127 and AHRI 550/590 or 551/591 define rating methods, not a universal COP interval. The grid is an effective plant-performance assumption pending a declared CRAH/chiller type and manufacturer map. |
| $T_{\mathrm{supply},A}$ (degC) | 20 **[O: energy-intensive]**; 27 **[O]**; 30 **[N]**; 32 **[B/NR with nominal $K_{\mathrm{liq}}$]** | Cold-water, W27, nominal, and upper W32 cases. The 32 degC point is standards/product allowed, but the current 35 MW/K conductance crossed the safety limit at continuous 150 MW even with maximum pumping. | ASHRAE liquid classes W27 and W32. Cray EX QuickSpecs specify facility-water supply up to 32 degC. A 40 degC case would require a separate W40-qualified architecture. |
| $T_{\mathrm{supply},B}$ (degC) | 18 **[O]**; 20 **[N]**; 24 **[O]**; 27 **[B: upper recommended inlet]** | Lower, nominal, intermediate, and upper recommended air-temperature cases. The 27 degC point is conditional because the benchmark variable is CRAH supply rather than measured equipment inlet. | ASHRAE A1 to A4 recommended equipment-inlet range of 18 to 27 degC. The comparison assumes inlet equivalence until a supply-to-inlet rise is specified. |
| $f_{\mathrm{fault}}$ | 1.0 **[N]**; 0.8 **[S/NR]**; 0.6 **[S/NR]**; 0.4 **[S/NR]** | Zero, 20%, 40%, and 60% aggregate conductance loss. All three degraded cases crossed 35 degC under full nameplate and maximum cooling. | This is not a material specification. A value of 0.8 can represent one of five equivalent units unavailable, although a properly sized N+1 plant should preserve nameplate capacity. Values 0.6 and 0.4 are compound or severe stress cases. The current implementation applies one factor to both zones. |
| $T_{\mathrm{amb}}$ (degC) | 25 **[N]**; 30 **[O: hot]**; 35 **[B: severe heat]**; 40 **[S: extreme heat]** | Nominal through extreme-hot weather. These values did not independently make the maximum-cooling plant infeasible, but 30 to 40 degC produced warning excursions in all closed-loop controllers. | Exogenous weather/scenario input. The 40 degC point corresponds to the paper's Thermal Squeeze scenario. |

The liquid-side base COP is excluded from this sweep: it changes Zone A
cooling electricity but not the Zone A temperature update, and belongs to a
separate power, tracking, and economic sensitivity experiment. IT heat input
is an experimental load driver: open-loop runs hold 150 MW in Zone A and
100 MW in Zone B; closed-loop runs use the benchmark workload trace.

## Interpretation of Capacitance

Capacitance controls transient inertia but cancels from the steady-state
solution. The relevant time constant is

$$
\tau_z=\frac{C_z}{K_{\mathrm{cool},z}+K_{\mathrm{env},z}}.
$$

At nominal full-cooling conductance, the selected grids cover approximately 6
to 19 minutes in each zone. The sweep therefore probes time-to-danger and
thermal-buffer sensitivity, not four named material substitutions.

## Interpretation of Cooling Conductance

At full Zone A nameplate and nominal supply/ambient conditions, the selected
liquid conductances span the feasible boundary:

| $K_{\mathrm{liq}}$ (MW/K) | Approximate Zone A equilibrium |
| ---: | ---: |
| 18.8 | 37.6 degC, infeasible at full load |
| 25.1 | 35.8 degC, near or above the safety boundary |
| 35.0 | 34.2 degC, nominally feasible |
| 50.1 | 32.9 degC, higher headroom |

The air-side values similarly represent alternative effective temperature
differences at 100 MW; reported results include the equilibrium temperature
and time constant so each aggregate value stays physically interpretable.

## Coupled Cases

| Case | Status | Parameters | Purpose |
| --- | --- | --- | --- |
| Nominal | **N** | Current benchmark values | Reference case |
| Low inertia | **B** | $C_A=13{,}500$, $C_B=5{,}000$ MJ/K | Tests reduced thermal buffering without changing steady-state feasibility |
| Weak cooling | **NR** | $K_{\mathrm{liq}}=25.1$, $K_{\mathrm{air}}=10.0$ MW/K | Tests reduced installed/active heat-removal conductance; Zone A is infeasible at continuous nameplate |
| Partial outage | **S/NR** | $f_{\mathrm{fault}}=0.8$ | Interpretable shared 20% conductance loss under the current implementation; not a recommended continuous operating condition |
| Combined adverse | **S/NR** | Low inertia; weak cooling; $T_{\mathrm{supply},A}=32$ degC; $T_{\mathrm{supply},B}=27$ degC; air COP 2.5; ambient 40 degC | Probes a compound emergency boundary without double-counting an additional fault multiplier |

Combining the lowest conductances with $f_{\mathrm{fault}}=0.4$ is omitted
from the main sweep: it compounds two representations of cooling loss and
would be retained only as an explicitly infeasible stress test.

## Experimental Setup

Run from the repository root with the project virtual environment active:

```powershell
python -m c2g_env.experiments.thermal_sensitivity
```

The completed experiment used the following setup:

| Item | Setting |
| --- | --- |
| Parameter configurations | 48: one nominal row, 43 OAT grid rows, and four named coupled cases. OAT grids retain their nominal value to make each per-parameter sequence self-contained, so some rows are physically duplicate reference points. |
| Horizon | 17,280 ticks, corresponding to 24 hours at 5 s per tick |
| Random seed | 100 |
| Initial zone temperature | $\min(T_{\mathrm{amb}}+5,34)$ degC, matching the benchmark environment warm start |
| Safety anchors | Warning above 33 degC; thermal termination above 35 degC |
| Open-loop heat input | Constant 150 MW Zone A and 100 MW Zone B, representing continuous full nameplate |
| Closed-loop heat input | Alibaba-derived benchmark workload trace through `WorkloadOrchestrator` |
| Closed-loop controllers | Frozen bang-bang, PID, and rule-based hardware controllers with their published 30/31/32.5/33/35 degC references unchanged |
| Closed-loop nonthermal state | Regulation command fixed to zero; SOC, voltage, frequency, and backlog observations held at benign values to isolate thermal feedback |
| Output | 288 rows in `c2g_env/experiments/thermal_tests_raw/thermal_sensitivity_full.csv` |

Each plant configuration receives three open-loop runs at full nameplate:

1. Minimum pump and zero HVAC (`open_worstcool`).
2. Pump and HVAC at 0.7 (`open_nominal`).
3. Maximum pump and HVAC (`open_maxcool`).

It also receives three closed-loop runs, one per frozen controller, against
the same workload trace. The output reports maximum and final temperatures,
warning-violation rate, headroom, first crossing time above 35 degC,
terminating zone, and mean actions. Each is one deterministic thermal
trajectory per configuration and controller, not a five-seed scenario
evaluation.

The harness continues past the first crossing to characterize the full
24-hour trajectory; `crossed_tsafe=1` marks a row where `C2GFastEnv` would
have terminated. It does not simulate BESS, grid-frequency, PCC-voltage, or
backlog termination, so it cannot produce or rule out nonthermal
terminations; those require a full-environment run.

## Results: Normal Operating Coefficients

The normal subset contains values labelled **N** or **O**, excluding
low-inertia boundaries, low-headroom cooling, degraded fault factors, extreme
ambient temperatures, and the coupled stress cases. Across 28 distinct
normal/reference rows, maximum cooling at continuous nameplate produced no
thermal termination; the highest open-loop temperatures were 34.272 degC
(Zone A) and 31.444 degC (Zone B).

The corresponding 84 frozen-controller runs on the workload trace produced no
warning excursions above 33 degC and no thermal terminations:

| Controller result over normal/reference rows | Value |
| --- | ---: |
| Controller runs | 84 |
| Warning-level rows | 0 |
| Thermal terminations | 0 |
| Highest Zone A temperature | 31.031 degC |
| Highest Zone B temperature | 29.973 degC |

The published analytic controllers thus retain substantial thermal headroom
under the observed workload with ordinary or nominal coefficients. This
partly reflects the workload's operating point and should not be read as
evidence of safety at continuous 250 MW nameplate.

The nominal plant illustrates the distinction: the full-nameplate run remains
feasible under maximum cooling, but with both cooling commands fixed at 0.7,
Zone A crosses 35 degC after 2,030 s (33.8 min). Nominal thermal coefficients
therefore do not make safety automatic under full load; sufficient cooling
effort remains necessary.

## Results: Stress Testing

The controller-free maximum-cooling test identified nine failing
configurations, all first crossing in Zone A:

| Configuration | Status | First crossing | Interpretation |
| --- | --- | ---: | --- |
| $K_{\mathrm{liq}}=18.8$ MW/K | **NR** | 24.75 min | Under-sized for continuous 150 MW |
| $K_{\mathrm{liq}}=25.1$ MW/K | **NR** | 35.50 min | Near-boundary but still under-sized |
| $T_{\mathrm{supply},A}=32$ degC | **B/NR** | 21.42 min | W32 boundary is allowed, but not feasible with nominal conductance at 150 MW |
| $f_{\mathrm{fault}}=0.8$ | **S/NR** | 53.42 min | Plausible 20% degradation; marginally below required effective conductance |
| $f_{\mathrm{fault}}=0.6$ | **S/NR** | 27.25 min | Severe shared 40% conductance loss |
| $f_{\mathrm{fault}}=0.4$ | **S/NR** | 20.92 min | Major shared 60% conductance loss |
| `weak_cooling` | **NR** | 35.50 min | Coupled case containing $K_{\mathrm{liq}}=25.1$ and $K_{\mathrm{air}}=10$ |
| `partial_outage` | **S/NR** | 53.42 min | Named counterpart of $f_{\mathrm{fault}}=0.8$ |
| `combined_adverse` | **S/NR** | 2.50 min | Compound emergency, warm-started at 34 degC because ambient is 40 degC |

`weak_cooling` and `partial_outage` intentionally overlap their OAT
counterparts, showing the same adverse mechanism inside a named facility case
rather than compounding two independent failures.

For context, minimum cooling caused all 48 configurations to cross 35 degC
under full nameplate; fixed 0.7 cooling caused 45 of 48 to cross. Only
$K_{\mathrm{liq}}=50.1$ MW/K and Zone A supply temperatures of 20 or 27 degC
avoided termination at 0.7 effort. These are feasibility controls, not
recommended operating schedules.

None of the 144 frozen-controller runs on the workload trace terminated,
including the stress cases. Warning excursions occurred only for ambient
temperatures of 30, 35, and 40 degC and for `combined_adverse`: ambient-only
warning fractions stayed between 0.29% and 0.48%, while `combined_adverse`
reached 19.65% (bang-bang), 100% (PID), and 22.38% (rule-based), with peak
temperatures nevertheless below 34 degC.

Together, the tests separate two findings: the controllers remain thermally
safe across the sampled workload trajectory, including substantial parameter
stress; and several practically interpretable boundary cases are not
physically feasible at continuous nameplate, even under maximum cooling. The
latter indicates a meaningful safety boundary in the simulator rather than
safety guaranteed by permissive coefficients.

No nonthermal termination occurred or could occur in this reduced harness,
which does not instantiate BESS depletion, frequency, PCC-voltage, or backlog
faults. Claims about those termination modes or cross-scenario survival
require the full four-scenario environment runs below.

## Full-Environment Cross-Scenario Evaluation

We subsequently evaluated the sensitivity grid in `C2GMacroEnv`, including the
thermal, BESS, grid-frequency, PCC-voltage, and SLA-backlog termination
paths, collapsing OAT rows that reproduce the nominal plant to 36 unique
configurations. The factorial design spans four scenarios, five seeds
(100 to 104), and four hardware controllers: 2,880 episodes.

The macro controller is held fixed as `RuleBasedMacroController`, a
deterministic policy that uses the same market and safety observations for
every hardware policy, limiting high-level policy variation as a confound.
The hardware policies are bang-bang, PID, rule-based, and the frozen Phase-1
SAC checkpoint at
`trained_models/BestCheckpoints/RuleHighlevel_SacLowlevel/final_model.zip`,
not retrained or tuned for any scenario or thermal configuration.

Thermal overrides are applied after scenario construction, and the runner
passes only values that differ from the nominal plant: Scenario B retains its
40 degC ambient condition and Scenario C its 0.6 cooling-fault factor unless
the evaluated case changes the same quantity. Each row records the five
termination flags from the final inner-environment state and accumulates
warning exposure over all fast substeps.

The table reports mean and sample standard deviation over five seed-level
fractions, each spanning the same 36 plant configurations; fault counts are
totals over 180 episodes per scenario-controller pair.

| Scenario | Hardware controller | Survival, % | Warning-step fraction, % | Thermal faults | Thermal faults under stress conditions | SLA faults |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Default | Bang-bang | 98.89 +/- 1.52 | 0.70 +/- 0.03 | 0 | 0 | 2 |
| Default | PID | 100.00 +/- 0.00 | 2.70 +/- 0.00 | 0 | 0 | 0 |
| Default | Rule-based | 100.00 +/- 0.00 | 0.56 +/- 0.00 | 0 | 0 | 0 |
| Default | SAC | 97.78 +/- 1.24 | 1.57 +/- 0.91 | 4 | 4 | 0 |
| Scenario A | Bang-bang | 98.89 +/- 1.52 | 1.08 +/- 0.03 | 0 | 0 | 2 |
| Scenario A | PID | 100.00 +/- 0.00 | 3.32 +/- 0.00 | 0 | 0 | 0 |
| Scenario A | Rule-based | 100.00 +/- 0.00 | 0.96 +/- 0.00 | 0 | 0 | 0 |
| Scenario A | SAC | 97.22 +/- 1.96 | 7.63 +/- 0.81 | 5 | 5 | 0 |
| Scenario B | Bang-bang | 98.89 +/- 1.52 | 1.08 +/- 0.04 | 0 | 0 | 2 |
| Scenario B | PID | 100.00 +/- 0.00 | 4.09 +/- 0.03 | 0 | 0 | 0 |
| Scenario B | Rule-based | 100.00 +/- 0.00 | 0.99 +/- 0.01 | 0 | 0 | 0 |
| Scenario B | SAC | 97.78 +/- 1.24 | 8.45 +/- 0.96 | 4 | 4 | 0 |
| Scenario C | Bang-bang | 97.78 +/- 3.04 | 3.83 +/- 0.39 | 0 | 0 | 4 |
| Scenario C | PID | 100.00 +/- 0.00 | 6.46 +/- 0.00 | 0 | 0 | 0 |
| Scenario C | Rule-based | 100.00 +/- 0.00 | 3.90 +/- 0.01 | 0 | 0 | 0 |
| Scenario C | SAC | 94.44 +/- 2.78 | 11.13 +/- 1.25 | 10 | 10 | 0 |

"Thermal faults under stress conditions" is the subset of "Thermal faults"
whose plant configuration has `stress_condition = True`: at least one swept
parameter at an **NR** grid value (defined below). Across all 16
scenario-controller pairs, every recorded thermal fault occurred under such a
configuration; none occurred under the nominal plant, an ordinary/reference
OAT variant, or the B-only `low_inertia` case.

The nominal plant survived all 80 scenario-controller-seed episodes without a
fault termination; its largest observed temperature was 33.994 degC. Default
nominal episodes never entered the 33 degC warning region, and nominal
warning-step fractions stayed below 1% in Scenarios A and B and below 3% in
Scenario C.

Across the full grid, PID and rule-based control survived all 720 episodes
per controller. Bang-bang had no thermal termination; its ten terminations
were SLA-backlog faults in `combined_adverse` and, under Scenario C,
`T_supply_A=32.0`. SAC had 23 thermal terminations: seventeen in
`combined_adverse`, and six more in `fault_factor=0.4` under Scenario A,
`K_liq=18.8` under Scenario C, and `T_supply_A=32.0` under Scenario C. No
controller produced a frequency, voltage, or SOC fault.

Every Scenario A, B, and C episode entered the warning region at least once,
though this exposure did not generally imply termination. Seed-mean
warning-step fractions ranged 0.96%-7.63% in Scenario A, 0.99%-8.45% in
Scenario B, and 3.83%-11.13% in Scenario C. SAC had the largest warning
exposure in each stress scenario and was the only controller with thermal
terminations, suggesting the frozen SAC policy is less robust to joint
scenario and plant shifts than the analytic controllers. The observed
failures nonetheless concern declared boundary or stress configurations, not
the nominal plant.

The merged episode output is stored in
`c2g_env/experiments/thermal_tests_raw/thermal_sensitivity_cross_scenario.csv`, with seed-level,
scenario-level, and configuration-level summaries alongside it under the
suffixes `_seed_summary`, `_summary`, and `_config_summary`.

The configuration-level summary (`_config_summary`) additionally reports a
`stress_condition` column, `True` only when at least one swept parameter
deviates from nominal onto a grid point annotated **NR** in the Sweep table
above. It is `True` for $K_{\mathrm{liq}}\in\{18.8,25.1\}$,
$T_{\mathrm{supply},A}=32$, $f_{\mathrm{fault}}\in\{0.8,0.6,0.4\}$, and the
coupled cases `weak_cooling`, `partial_outage`, and `combined_adverse`, each
of which touches at least one NR value. Deviations reaching only a **B**-only
value, an **S**-only value (currently $T_{\mathrm{amb}}=40$ degC), or an
ordinary/nominal value leave it `False`, as do `nominal`, all ordinary OAT
variants, and `low_inertia` (which touches only the B-labeled $C_A=13{,}500$
and $C_B=5{,}000$). The column lets downstream analysis condition survival,
warning, and fault totals on whether a configuration is itself not-recommended
rather than ordinary, nominal, or merely boundary.

## Source Families

1. NIST Chemistry WebBook and IAPWS-95 for water and air properties.
2. NIST-JANAF tables for iron, aluminum, and copper heat capacities.
3. ASHRAE Fundamentals and ISO 10456 for glycol coolants, concrete, building
  materials, and infiltration calculations.
4. Cray Supercomputing EX QuickSpecs (document a00094635enw, version 21) for
  CDU capacity, cabinet power, and water-temperature limits.
5. CoolIT CHx2000, CHx80, and AHx240 product specifications for alternative
  liquid-cooling equipment scales.
6. Vertiv Liebert CW specifications for CRAH capacities and airflow.
7. ASHRAE TC 9.9 thermal-guideline reference card for liquid classes and air
  inlet recommendations.
8. ASHRAE 90.1, IECC C402, and DOE-PNNL commercial prototype models for
  envelope assemblies, climate zones, geometry, and infiltration.
9. ASHRAE 127 and AHRI 550/590 or 551/591 for cooling-equipment rating methods.

## Claim Boundary

The experiment shows whether benchmark conclusions remain stable across
physically interpreted thermal cases and where the modeled plant becomes
infeasible. It does not validate the nominal aggregate parameters against a
specific facility; that requires a component inventory or measured thermal
step responses (temperature, IT power, supply conditions, flow/airflow,
cooling commands, electrical input) under a declared meter boundary.
