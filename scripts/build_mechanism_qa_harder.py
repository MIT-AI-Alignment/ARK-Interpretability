"""Build data/items/mechanism_qa_harder.jsonl: 40 mechanism-QA items in domains where
Qwen 2.5's training is likely shallow. Schema mirrors data/items/mechanism_qa.jsonl.

Item composition (40 total, 26 train / 14 eval, stratified by topic_bucket):
  - instrumentation       7 items (4 train / 3 eval)
  - manufacturing         7 items (5 train / 2 eval)
  - electrochemistry      6 items (4 train / 2 eval)
  - molecular_biology     7 items (5 train / 2 eval)
  - geophysics            7 items (4 train / 3 eval)
  - engineering           6 items (4 train / 2 eval)

Item ids are mechH_001..mechH_040. The "H" prefix distinguishes these from the original
80-item set so item ids never collide if both files are loaded together.

Selection rationale: items are chosen by an a priori obscurity proxy (specialty technical
domains rarely covered in instruction-tuning data), not by querying Qwen. We measure Qwen's
accuracy as the experimental output, never the filter — preserves IOED-style methodology
where the difficulty score is hidden from the subject.
"""

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_PATH = REPO / "data" / "items" / "mechanism_qa_harder.jsonl"
HASHES_PATH = REPO / "data" / "items" / "HASHES.txt"
MATH_PATH = REPO / "data" / "items" / "math.jsonl"
MECH_PATH = REPO / "data" / "items" / "mechanism_qa.jsonl"


ITEMS: list[dict] = [
    # ---------------- instrumentation (7) ----------------
    {
        "topic_bucket": "instrumentation",
        "question": "How does an Orbitrap mass analyzer determine the mass-to-charge ratio of an ion? Explain the mechanism step by step.",
        "reference_answer": [
            "Ions injected from a C-trap into the Orbitrap are radially trapped between an outer barrel electrode and a central spindle-shaped electrode by an electrostatic field; the field also confines them axially in a quadro-logarithmic potential well.",
            "Trapped ions oscillate harmonically along the central spindle's axis; the oscillation frequency is omega = sqrt(k/(m/z)), so frequency depends only on m/z (and the trap geometry constant k), independent of ion energy or initial position to first order.",
            "Coherent axial motion of the ion packet induces an image current on the two halves of the split outer electrode; the time-domain image current is digitized and Fourier-transformed to extract oscillation frequencies, which map to m/z values.",
            "Mass resolution scales with the acquisition time T (because frequency resolution scales as 1/T) and with the perfection of the electrode shape and voltage stability; resolving power above 100,000 at moderate m/z is routine, set by detector noise, residual gas pressure, and trap field aberrations.",
        ],
        "rubric_notes": "A shallow answer says ions are trapped and weighed in an electric field without explaining that the measured quantity is the axial oscillation frequency of the trapped ion packet (related to m/z by an inverse square root) extracted via image-current Fourier transform.",
    },
    {
        "topic_bucket": "instrumentation",
        "question": "How does dose fractionation in cryo-EM single-particle imaging recover high-resolution structural information that a single integrated exposure would lose? Explain the mechanism step by step.",
        "reference_answer": [
            "A direct electron detector records many short frames (typically tens of frames over a few seconds) capturing the same total electron dose as a single exposure but resolved in time.",
            "Beam-induced motion of the vitrified specimen is most rapid in the earliest frames (driven by radiolysis-induced charging and stage relaxation); per-frame alignment to a running average corrects this motion before integration, restoring high-spatial-frequency contrast that would be blurred in a single integrated frame.",
            "Radiation damage destroys high-spatial-frequency Fourier components first while preserving low-frequency signal longer; per-frequency exposure weighting (e.g., the Grant-Grigorieff scheme) gives early frames more weight at high resolution and later frames more weight at low resolution before recombination.",
            "Without fractionation, motion blur and dose-cumulative damage to high-frequency components irreversibly cap resolution at perhaps 6-10 angstrom on flexible specimens; with fractionation plus motion correction and exposure weighting, atomic-resolution maps below 3 angstrom are achievable.",
        ],
        "rubric_notes": "A shallow answer says fractionation reduces motion blur, missing the second half: dose-dependent radiation damage is frequency-selective, and per-frame exposure weighting recovers high-frequency information from early frames before it is destroyed.",
    },
    {
        "topic_bucket": "instrumentation",
        "question": "How does atomic force microscopy in tapping (intermittent-contact) mode produce a topographic image of a soft sample without destroying it? Explain the mechanism step by step.",
        "reference_answer": [
            "A cantilever with a sharp tip is driven near its mechanical resonance frequency by a piezo shaker; in free air the cantilever oscillates with a known free amplitude (tens to hundreds of nm).",
            "When brought near a surface, intermittent tip-sample contact damps the cantilever oscillation, reducing amplitude and shifting phase; the system feedbacks on amplitude to a setpoint by adjusting the z-piezo distance to the sample.",
            "Topography is recovered by recording the z-piezo position required to maintain the setpoint amplitude as the tip rasters x-y across the surface; lateral (frictional) forces are minimized because tip-sample contact is brief and predominantly normal.",
            "The phase lag between drive and response carries information about local viscoelastic dissipation (adhesion, stiffness), enabling material contrast separately from topography; sub-nanometer height resolution is set by the noise floor of the photodiode position sensor and thermal cantilever motion.",
        ],
        "rubric_notes": "A shallow answer says the tip taps the surface and a feedback loop tracks height, without explaining what is held constant (oscillation amplitude), why intermittent contact spares soft samples (lateral forces are nearly eliminated), or what phase information conveys.",
    },
    {
        "topic_bucket": "instrumentation",
        "question": "How does STED (stimulated-emission-depletion) microscopy achieve resolution below the diffraction limit? Explain the mechanism step by step.",
        "reference_answer": [
            "An excitation laser focused to a diffraction-limited spot promotes fluorophores within the spot from ground state to an excited electronic state.",
            "A second 'depletion' laser, shaped (e.g., by a vortex phase plate) into a doughnut with a true zero of intensity at the center, is overlaid on the excitation spot; its wavelength matches the fluorophore's stimulated-emission band.",
            "In the doughnut periphery, the depletion beam stimulates excited fluorophores back to the ground state, emitting at the depletion wavelength which is filtered out; only fluorophores at the doughnut zero survive long enough to fluoresce spontaneously and be detected.",
            "The effective resolution is the size of the residual emitting region at the doughnut zero; it scales as lambda/(2 NA sqrt(1 + I_dep/I_sat)), so increasing depletion intensity above saturation shrinks the emitting region without bound in principle. Practical limits are photobleaching, detector noise, and fluorophore choice (must have a usable stimulated-emission cross section at the depletion wavelength).",
        ],
        "rubric_notes": "A shallow answer says a second beam suppresses fluorescence outside the center without explaining that the depletion beam is doughnut-shaped with a zero at the center, that the suppression mechanism is stimulated emission (not bleaching), or that resolution scales with the square root of depletion intensity ratio above saturation.",
    },
    {
        "topic_bucket": "instrumentation",
        "question": "How does a surface plasmon resonance (SPR) sensor detect biomolecular binding without labeling the analyte? Explain the mechanism step by step.",
        "reference_answer": [
            "Polarized light is reflected off a thin gold film deposited on a glass prism; in the Kretschmann configuration the prism couples evanescent waves into the metal and at a specific incidence angle the photon momentum matches the surface plasmon polariton momentum at the metal-dielectric interface.",
            "At that resonance angle, energy is transferred from the photons into collective electron oscillations (surface plasmons) at the gold surface, and the reflectance shows a sharp dip; the resonance angle depends on the refractive index of the dielectric within the evanescent decay length (~hundreds of nm) above the gold.",
            "Ligand molecules immobilized on the gold surface capture analyte from a flowing buffer; binding adds mass at the interface and locally raises the refractive index, shifting the resonance angle (or the resonance wavelength in spectral SPR).",
            "Real-time tracking of the resonance shift versus time (a sensorgram) reports association and dissociation kinetics directly; this is label-free because no fluorophore or radiolabel is required, only a refractive-index change near the surface.",
        ],
        "rubric_notes": "A shallow answer says SPR detects binding by an optical signal change without identifying the resonance condition (angle/wavelength matching photon momentum to surface plasmon momentum) or what the measurement actually tracks (the refractive-index-induced shift of the resonance angle, not absolute light intensity).",
    },
    {
        "topic_bucket": "instrumentation",
        "question": "How does time-of-flight mass spectrometry with a reflectron determine m/z, and why does the reflectron improve resolution? Explain the mechanism step by step.",
        "reference_answer": [
            "Ions formed at the source are accelerated through a fixed potential difference V, gaining kinetic energy zeV; in the field-free drift region of length L, an ion of mass m has velocity v = sqrt(2zeV/m), so heavier ions are slower and arrive later at the detector.",
            "Arrival time t = L/v is proportional to sqrt(m/z); converting time-of-flight to m/z requires only the calibration of the geometry and accelerating voltage (using known reference ions).",
            "Real ions of identical m/z have a small spread in initial kinetic energy and position at the source; this translates directly into a spread in flight time, broadening the mass peak and limiting resolution in a simple linear TOF.",
            "A reflectron (electrostatic ion mirror) at the end of the drift region decelerates and reflects ions back; ions with slightly higher kinetic energy penetrate deeper into the mirror and spend longer being turned around, while lower-energy ions turn faster, so all ions of the same m/z exit the mirror at the same time. This first-order energy compensation sharpens peaks substantially, improving resolution.",
        ],
        "rubric_notes": "A shallow answer says heavier ions take longer to fly and gives the formula but omits the source of resolution loss (kinetic-energy spread at the source) and how the reflectron compensates by making higher-energy ions take a longer path inside the ion mirror.",
    },
    {
        "topic_bucket": "instrumentation",
        "question": "How does a confocal fluorescence microscope produce optical sectioning that a conventional widefield microscope cannot? Explain the mechanism step by step.",
        "reference_answer": [
            "Excitation light is focused to a diffraction-limited spot inside the sample by the objective; fluorophores in a small focal volume are excited and emit isotropically.",
            "Emitted fluorescence is collected by the same objective and re-imaged onto a pinhole placed at the conjugate image plane in the detection arm; in-focus emission converges to the pinhole and passes through to the detector (PMT or APD), while emission from above and below the focal plane forms a defocused blur at the pinhole and is largely blocked.",
            "The illumination point is scanned in x and y (galvo mirrors or stage motion), and the detector signal at each scan position is assembled into an image; moving the focus in z and repeating produces a 3D stack with axial sectioning of about 500 nm depending on NA and pinhole diameter.",
            "Smaller pinhole gives thinner sections and better z-resolution but lower signal throughput; widefield microscopy lacks this rejection mechanism, so out-of-focus fluorescence everywhere in the illuminated volume reaches the camera and hazes any single in-focus plane.",
        ],
        "rubric_notes": "A shallow answer says a confocal scope uses a pinhole to make sharper images without explaining that the pinhole's optical role is rejecting fluorescence from out-of-focus planes by virtue of being at the conjugate focal plane, which is what produces axial sectioning.",
    },

    # ---------------- manufacturing (7) ----------------
    {
        "topic_bucket": "manufacturing",
        "question": "How does the float glass (Pilkington) process produce flat sheet glass with optical-quality surfaces? Explain the mechanism step by step.",
        "reference_answer": [
            "Molten glass at about 1100 C is delivered continuously from the furnace onto the surface of a long bath of molten tin (~700-1000 C across the bath); tin's higher density (about 6.5 g/cm^3 vs. about 2.5 g/cm^3 for glass) supports the glass as it spreads.",
            "The glass spreads under gravity to its natural equilibrium thickness (~6.5 mm) determined by surface tension between glass, tin, and atmosphere; edge rollers (top rollers) can stretch the ribbon thinner or compress it thicker than the equilibrium value.",
            "The bottom of the ribbon contacts a perfectly flat liquid-tin surface (no nucleation sites or friction), and the top is the free upper surface, also flat under gravity; both surfaces solidify as the ribbon cools while traveling along the bath, producing two optical-quality flat surfaces simultaneously.",
            "The bath sits under a sealed reducing atmosphere (typically N2 with a few percent H2) to prevent SnO2 formation, which would adhere to the glass; trace tin diffusion into the bottom surface still leaves a faint asymmetry detectable by UV fluorescence (the 'tin side'), important for downstream coating.",
        ],
        "rubric_notes": "A shallow answer says glass floats on tin and cools flat without explaining why both surfaces become flat (one against the perfectly flat liquid surface, the other free under gravity) or why the atmosphere must be reducing (to prevent SnO2 adhesion).",
    },
    {
        "topic_bucket": "manufacturing",
        "question": "How does the Bayer process separate alumina from bauxite ore in industrial aluminum production? Explain the mechanism step by step.",
        "reference_answer": [
            "Crushed bauxite is digested in hot concentrated NaOH (~150-250 C, elevated pressure) inside autoclaves; aluminum-bearing phases (gibbsite Al(OH)3, boehmite AlOOH, diaspore AlOOH) dissolve as soluble sodium aluminate via Al(OH)3 + NaOH -> Na+ + Al(OH)4-.",
            "Iron oxides, silica, titanium dioxide, and other impurities remain insoluble and are separated as red mud by settling and filtration; the clarified caustic liquor now contains aluminum selectively pulled out of the ore.",
            "The clarified liquor is cooled and seeded with fine gibbsite crystals; supersaturation reverses the equilibrium and Al(OH)3 nucleates on the seeds, growing into precipitate that is separated by filtration. The remaining caustic solution is reconcentrated and recycled.",
            "Recovered Al(OH)3 is washed and calcined at about 1000 C in a kiln, dehydrating it to alpha-Al2O3 (alumina); this alumina powder is the feedstock for the Hall-Heroult electrolytic cell that produces metallic aluminum.",
        ],
        "rubric_notes": "A shallow answer says caustic dissolves alumina and the rest is filtered without explaining the cyclic supersaturation/seeding step that re-precipitates aluminum hydroxide, or the final calcination step that dehydrates the hydroxide to alumina.",
    },
    {
        "topic_bucket": "manufacturing",
        "question": "How does metal-organic chemical vapor deposition (MOCVD) grow a single-crystal compound semiconductor layer (e.g., GaN) on a substrate? Explain the mechanism step by step.",
        "reference_answer": [
            "Volatile metal-organic precursors (e.g., trimethylgallium, TMGa) and a hydride (e.g., NH3) are flowed in carrier gas over a heated single-crystal substrate (e.g., sapphire) inside a high-purity reactor at hundreds of Torr.",
            "Near the hot substrate surface, gas-phase reactions and surface decomposition of the precursors release Ga adatoms, NH-related species, and methyl byproducts; reactant transport is by diffusion across a stagnant boundary layer above the substrate.",
            "Adatoms adsorbed on the surface diffuse and incorporate preferentially at step edges and kinks, in registry with the underlying lattice; this two-dimensional step-flow growth produces an epitaxial single-crystal layer atom-by-atom.",
            "Substrate temperature, V/III ratio, and growth rate must balance: high enough temperature for adatom diffusion to reach kinks, high enough V/III to suppress vacancies, low enough rate to avoid 3D islanding; lattice mismatch with the substrate is accommodated by an initial low-temperature buffer or threading dislocations relax to the surface.",
        ],
        "rubric_notes": "A shallow answer says gases react and deposit a film without explaining that single-crystal growth requires adatom surface diffusion to step/kink sites in registry with the substrate lattice, or the role of the V/III ratio and temperature window.",
    },
    {
        "topic_bucket": "manufacturing",
        "question": "How does the Czochralski process pull a single-crystal silicon ingot from a melt? Explain the mechanism step by step.",
        "reference_answer": [
            "High-purity polycrystalline silicon is melted in a quartz crucible inside an inert-atmosphere chamber; a small dislocation-free single-crystal seed (with the desired orientation, e.g., <100>) is dipped into the melt surface from above.",
            "The seed is rotated and slowly withdrawn upward; melt at the moving solid-liquid interface solidifies onto the seed, copying its lattice orientation, so the rod growing above the melt is single-crystal silicon.",
            "After dipping, the seed is necked down to a narrow diameter (Dash neck); thermal stress dislocations created at the seed-melt interface preferentially propagate to the side surface in the narrow neck and exit the crystal, leaving a dislocation-free continuation that is then expanded to the full ingot diameter.",
            "Counter-rotation of the crucible and ingot, controlled thermal gradients, and applied magnetic fields (MCZ) damp melt convection and segregation; pull rate and rotation are servo-controlled to maintain a stable solid-liquid interface and a uniform doping profile along the boule.",
        ],
        "rubric_notes": "A shallow answer says a seed is dipped and pulled to grow a single crystal without explaining the Dash neck for dislocation removal, the role of crucible/ingot counter-rotation in suppressing convection, or the relation between pull rate and the moving solid-liquid interface.",
    },
    {
        "topic_bucket": "manufacturing",
        "question": "How does the chemical vapor infiltration (CVI) process densify a porous fibrous preform into a carbon-carbon composite? Explain the mechanism step by step.",
        "reference_answer": [
            "A fibrous carbon preform (e.g., woven carbon fabric or felt) is placed in a hot reactor; a hydrocarbon precursor gas (methane, propane, or natural gas) is admitted at low pressure with the preform held at about 1000 C.",
            "The hydrocarbon diffuses into the preform's open porosity and pyrolyzes within the pores, depositing solid pyrolytic carbon onto fiber surfaces; the deposited matrix gradually fills the void space.",
            "Reactant transport into the interior is diffusion-limited: the outer surface densifies fastest and tends to seal off the interior. Practical CVI cycles include intermediate machining of the surface skin to reopen access, sometimes with thermal/pressure gradients (thermal-gradient CVI, pulse CVI) used to drive deposition outside-in or inside-out.",
            "Cycles of infiltration and machining are repeated for hundreds of hours until target density is reached (often 1.6-1.9 g/cm^3); the resulting carbon-carbon composite has carbon fibers in a pyrolytic carbon matrix and is used in rocket nozzles and aircraft brake disks for high-temperature performance.",
        ],
        "rubric_notes": "A shallow answer says hydrocarbon gas decomposes inside the preform to deposit carbon without explaining that diffusion-limited transport causes outer-skin sealing, requiring intermediate machining or thermal-gradient strategies to densify the interior.",
    },
    {
        "topic_bucket": "manufacturing",
        "question": "How do organic leveling agents in an electroplating bath produce a smooth deposit on a rough cathode? Explain the mechanism step by step.",
        "reference_answer": [
            "Without additives, current density is highest at protrusions on a rough cathode (the field crowds at sharp features); metal deposition is therefore faster at peaks than in valleys, amplifying roughness.",
            "Leveling agents (e.g., thiourea derivatives, polyethers, certain dyes) are organic molecules that adsorb on the cathode surface; their adsorption is mass-transport-limited and therefore is highest at high-current-density peaks where the diffusion layer is thinner.",
            "Adsorbed leveler locally inhibits metal deposition (raising the local overpotential), redirecting current to recessed valley regions and slowing growth at peaks; net effect is more uniform deposition rate across the surface, so peaks grow no faster than valleys and roughness decreases.",
            "Leveler is consumed (incorporated into the deposit or destroyed) during plating, requiring continuous replenishment; bath analytics (cyclic voltammetric stripping, CVS) monitor accelerator/suppressor/leveler balance, since an out-of-balance bath fails to level or produces brittle deposits.",
        ],
        "rubric_notes": "A shallow answer says additives smooth the deposit without explaining the mechanism: leveler adsorption is mass-transport-driven, preferentially blocks peak sites where diffusion is faster, and redirects current density to valleys.",
    },
    {
        "topic_bucket": "manufacturing",
        "question": "How does pressureless sintering convert a green powder compact into a dense polycrystalline ceramic or metal part? Explain the mechanism step by step.",
        "reference_answer": [
            "A pressed compact of fine powder (the green body) consists of touching particles with high specific surface area and intervening voids; it is heated below the melting point in a controlled atmosphere or vacuum.",
            "The driving force is reduction of total surface (and grain boundary) free energy: atoms migrate from convex surfaces (high chemical potential) to concave neck regions between particles (low chemical potential), growing necks and reducing surface area.",
            "Several diffusion paths contribute: surface diffusion and evaporation-condensation move atoms but do not densify (no center-to-center approach), while grain-boundary and volume diffusion do densify by transporting atoms from the contact area outward and pulling particles closer together.",
            "As necks merge, isolated pores form and shrink (driven by Laplace pressure across curved surfaces); pore-grain boundary attachment is critical: if grains grow faster than pores can shrink, pores detach and become trapped intragranularly, limiting final density. Atmosphere choice (e.g., reducing for metals, inert/oxidizing for ceramics) prevents undesired surface reactions.",
        ],
        "rubric_notes": "A shallow answer says heat causes powder particles to fuse without explaining that densification requires diffusion paths that move particle centers closer (grain-boundary and volume diffusion), or that pore-boundary detachment by abnormal grain growth limits final density.",
    },

    # ---------------- electrochemistry (6) ----------------
    {
        "topic_bucket": "electrochemistry",
        "question": "How does the solid electrolyte interphase (SEI) form on a graphite anode in a lithium-ion cell, and why is it necessary? Explain the mechanism step by step.",
        "reference_answer": [
            "On first charge, Li+ intercalates into graphite at potentials very close to lithium metal (about 0.1 V vs Li/Li+); this is far below the reduction stability window of common carbonate electrolyte solvents (EC, DMC, EMC), so the electrolyte itself reduces at the graphite surface.",
            "Reduction of solvent and salt produces a complex layer of lithium alkyl carbonates, Li2CO3, LiF (from LiPF6 salt decomposition), and oligomeric/polymeric organics that precipitate on the graphite surface to form a thin (~tens of nm) film.",
            "Once formed, the SEI is electronically insulating but Li+-conductive: it blocks further electron transfer to electrolyte (passivating the anode and stopping further electrolyte reduction) while still allowing Li+ to shuttle in and out of graphite during cycling.",
            "SEI is dynamic: volume changes during intercalation crack the layer, exposing fresh graphite that triggers more electrolyte reduction; the SEI thickens, capacity fades, and impedance rises with cycling. Engineered electrolyte additives (FEC, VC) form more robust SEI components that suppress this growth and extend cycle life.",
        ],
        "rubric_notes": "A shallow answer says SEI is a protective layer formed during first charge without explaining that the protection mechanism is electronic insulation while remaining ionically conductive, and that the source of SEI is reduction of electrolyte at potentials below its stability window.",
    },
    {
        "topic_bucket": "electrochemistry",
        "question": "Why does pitting corrosion of stainless steel initiate in chloride-containing environments, and why does it self-sustain once started? Explain the mechanism step by step.",
        "reference_answer": [
            "Stainless steel's corrosion resistance comes from a thin (a few nm) self-healing chromium-rich oxide passive film; in oxygenated water this film repassivates after damage. Cl- ions, however, can adsorb at film defects and at metallurgical heterogeneities (especially MnS inclusions) and locally disrupt the film.",
            "At the disrupted site, exposed metal dissolves anodically: Fe -> Fe2+ + 2e-, while reduction of dissolved O2 occurs over the surrounding passive surface (the cathode); the small anode and large cathode produce intense local current density.",
            "Inside the developing pit, cation hydrolysis (Fe2+ + 2 H2O -> Fe(OH)2 + 2 H+) acidifies the solution while electromigration concentrates Cl- to balance accumulating cations; local pH can drop below 1 and Cl- concentrations exceed bulk by orders of magnitude.",
            "The acidic chloride microenvironment prevents repassivation and accelerates dissolution; the pit becomes self-sustaining, growing deep and narrow under a cathodically protected outer surface. Mo additions (316 grade) help by forming molybdate species that stabilize the pit interior chemistry against breakdown.",
        ],
        "rubric_notes": "A shallow answer says chlorides break the passive film without explaining the autocatalytic pit chemistry: cation hydrolysis acidifies the pit, electromigration concentrates Cl-, and the acidic-chloride pit interior is what prevents repassivation.",
    },
    {
        "topic_bucket": "electrochemistry",
        "question": "Why does combined tensile stress and a specific chemical environment cause stress-corrosion cracking in a metal that would resist either factor alone? Explain the mechanism step by step.",
        "reference_answer": [
            "Specific metal-environment pairs (austenitic stainless in hot Cl-, alpha-brass in NH3, mild steel in caustic, high-strength steels in H-containing environments) show stress-corrosion cracking; the same metal in distilled water under the same load, or unstressed in the same environment, does not crack.",
            "At a stressed surface defect or grain boundary, plastic strain ruptures the protective oxide/passive film and exposes fresh metal; the active environment dissolves the exposed metal rapidly before it can repassivate, advancing the crack tip by anodic dissolution.",
            "Alternatively (or in addition), the cathodic side reaction generates atomic hydrogen that diffuses ahead of the crack tip into the lattice, embrittling it; under stress, the embrittled region cracks at lower applied load (hydrogen embrittlement mechanism).",
            "Stress concentrates at the crack tip, accelerating film rupture (or hydrogen enrichment) at the tip while the bulk surface remains passive; the autocatalytic crack-tip cycle (rupture -> dissolution/embrittlement -> tip extension -> new rupture) cannot self-sustain without both factors. Mitigations: stress relief, environment control, alloy selection (resistant compositions).",
        ],
        "rubric_notes": "A shallow answer says SCC needs both stress and a corrosive environment without explaining the crack-tip cycle: stress ruptures the passive film, environment prevents repassivation, and either anodic dissolution or hydrogen embrittlement advances the crack at the tip.",
    },
    {
        "topic_bucket": "electrochemistry",
        "question": "Why do dendrites form on lithium-metal anodes during plating, and why are they dangerous in cells? Explain the mechanism step by step.",
        "reference_answer": [
            "On a lithium-metal anode, Li+ from electrolyte plates onto the surface; plating rate is set by local current density, which is non-uniform because of small surface roughness, SEI heterogeneity, and concentration gradients in the diffusion layer.",
            "At a small protrusion, current density is locally amplified (geometric field concentration) and Li+ flux is enhanced; the protrusion grows faster than its surroundings, becoming sharper and concentrating field even more (autocatalytic positive feedback).",
            "If the bulk current density approaches the limiting current (Sand's time analysis), Li+ near the electrode is depleted and the anion concentration profile drives space-charge effects that accelerate dendritic growth; non-uniform SEI coverage further localizes high-current spots.",
            "Once a dendrite penetrates the porous separator, it electrically shorts the anode to the cathode; large local current causes resistive heating, electrolyte decomposition, and potential thermal runaway. Mitigations: high-modulus solid electrolytes (mechanical block), 3D current collectors, high-concentration electrolytes (raise Li+ transference number), engineered SEI with FEC/LiF.",
        ],
        "rubric_notes": "A shallow answer says dendrites grow because plating is uneven without explaining the autocatalytic feedback (sharp protrusions concentrate current, accelerating their own growth), the role of approaching the diffusion-limited current, or why the danger is internal short-circuit causing thermal runaway.",
    },
    {
        "topic_bucket": "electrochemistry",
        "question": "Why does ppm-level CO in the hydrogen feed of a PEM fuel cell collapse anode performance, and what mitigations restore it? Explain the mechanism step by step.",
        "reference_answer": [
            "At a Pt anode, the hydrogen oxidation reaction proceeds via dissociative chemisorption (H2 + 2 Pt -> 2 Pt-H) followed by oxidation (Pt-H -> Pt + H+ + e-); high HOR rate requires a substantial fraction of free Pt sites available to bind H2.",
            "CO chemisorbs on Pt much more strongly than H (a typical adsorption-energy difference of tens of kJ/mol), so even a few ppm of CO in the H2 stream displaces adsorbed H and occupies most surface Pt sites at typical PEM operating temperatures (70-80 C).",
            "With Pt surface blocked by CO, the HOR rate per unit area collapses; the anode overpotential rises sharply and the cell voltage drops, even though hydrogen is plentiful in the gas phase. This is reversible if CO is removed, but irreversible degradation can occur if subsequent water management or carbon support corrosion follows.",
            "Mitigations: PtRu alloy catalysts (Ru oxidizes CO to CO2 at much lower potential than Pt, freeing Pt sites via the bifunctional mechanism), bleeding O2 into the anode (oxidizes CO directly), using high-temperature membranes (CO desorbs faster), or upstream feed cleanup (PrOx, methanation, PSA).",
        ],
        "rubric_notes": "A shallow answer says CO poisons the catalyst without quantifying why ppm levels matter (chemisorption energy gap displaces H from Pt) or explaining the bifunctional mechanism by which Ru in PtRu rescues activity.",
    },
    {
        "topic_bucket": "electrochemistry",
        "question": "How does sulfuric acid anodizing of aluminum produce a self-organized porous oxide layer, and what determines pore geometry? Explain the mechanism step by step.",
        "reference_answer": [
            "Al immersed in dilute H2SO4 at modest temperature is anodically polarized; at the metal-oxide interface, O2- migrates through existing oxide under the high field and forms new Al2O3, growing the oxide inward into the metal.",
            "Simultaneously, the acidic electrolyte dissolves the oxide at the oxide-electrolyte interface (field-assisted dissolution); a steady state arises in which the rate of new oxide formation equals the rate of dissolution at the surface, producing a thickening porous layer.",
            "Local field concentration at the bottom of nascent pores accelerates dissolution and metal-oxide interface advance there, deepening the pores; mechanical stress from volume mismatch between Al metal and Al2O3, plus electrostatic repulsion between adjacent pore bottoms, organizes the pores into a hexagonal pattern.",
            "Pore diameter and inter-pore spacing scale linearly with applied voltage (about 2.5 nm/V for sulfuric acid); pore depth grows linearly with time at constant current. Pores can subsequently be sealed by hydrothermal treatment (forming pseudoboehmite) for corrosion protection or dyed before sealing for decorative coloring; the same process is used industrially as a nanoporous template.",
        ],
        "rubric_notes": "A shallow answer says anodizing thickens the oxide layer without explaining the steady state between field-driven oxide growth and field-assisted dissolution at the pore bottom, or how voltage controls pore geometry.",
    },

    # ---------------- molecular_biology (7) ----------------
    {
        "topic_bucket": "molecular_biology",
        "question": "How does the kinesin-1 motor walk processively along a microtubule, and what prevents it from falling off after each step? Explain the mechanism step by step.",
        "reference_answer": [
            "Kinesin-1 is a homodimer of motor heads connected by a coiled-coil neck; each head has an ATPase nucleotide-binding pocket and a microtubule-binding interface, and the two heads are connected through flexible neck linkers (~14 amino acids).",
            "Walking is hand-over-hand toward the microtubule plus end: the rear head detaches from the microtubule, swings forward by 16 nm past the still-bound front head (driven by ATP-binding-induced docking of the front head's neck linker forward), and rebinds at the next tubulin site; then the roles reverse.",
            "The two heads are coordinated by mechanical and chemical gating: ATP hydrolysis (and thus head detachment) at the front head is suppressed until the rear head detaches and rebinds, ensuring at least one head is always microtubule-bound. The neck linker tension transmits this gating between heads.",
            "Run length is typically about 100 steps (about 1 micrometer) before stochastic dual detachment; load, ATP concentration, and salt all affect processivity. Gating mutations that decouple the head-head communication reduce processivity to a few steps because both heads can detach simultaneously.",
        ],
        "rubric_notes": "A shallow answer says kinesin walks hand-over-hand without explaining the mechanical/chemical gating between heads (front-head ATPase suppression while rear is bound) that ensures at least one head stays microtubule-attached, which is what makes the motor processive.",
    },
    {
        "topic_bucket": "molecular_biology",
        "question": "Why do synonymous codons translate at different rates, and what consequences does this have for protein expression? Explain the mechanism step by step.",
        "reference_answer": [
            "The genetic code is degenerate: most amino acids are encoded by two to six synonymous codons, which differ in their anticodon-pairing partners and thus in which tRNA isoacceptors deliver them.",
            "Cellular tRNA isoacceptor abundances vary widely; codons read by abundant tRNAs are decoded quickly because cognate aminoacyl-tRNA arrives at the ribosomal A site sooner, while codons read by rare tRNAs cause ribosome pausing while the ribosome waits for delivery.",
            "Local translation rate variation has functional consequences: pauses at rare-codon stretches can permit co-translational folding of preceding domains, support signal-recognition-particle binding for secretion, or bias mRNA stability via no-go decay pathways.",
            "Heterologous protein expression often fails when the source organism's codon usage mismatches the host's tRNA pool; codon optimization or supplementation with tRNAs for rare codons typically rescues yield. Recent work also shows codon-pair effects and correlations with protein abundance that go beyond single-codon usage.",
        ],
        "rubric_notes": "A shallow answer says different codons translate at different speeds because they bind different tRNAs without connecting tRNA isoacceptor abundance to ribosomal pause durations, or explaining downstream consequences like co-translational folding and heterologous expression yield.",
    },
    {
        "topic_bucket": "molecular_biology",
        "question": "How does the nuclear pore complex achieve selective transport, allowing only cargo with appropriate signal sequences while excluding bulk macromolecules? Explain the mechanism step by step.",
        "reference_answer": [
            "The nuclear pore complex (~120 MDa) spans the nuclear envelope and contains intrinsically disordered nucleoporins decorated with phenylalanine-glycine (FG) repeats; these FG repeats fill the central channel, forming a hydrophobic, weakly cohesive mesh.",
            "Small molecules (<~40 kDa) diffuse through the FG mesh passively; larger hydrophilic macromolecules cannot, because they cannot disrupt the hydrophobic interactions between FG repeats. The mesh is a soft selectivity filter, not a fixed pore.",
            "Cargo destined for the nucleus is bound by a karyopherin/importin (e.g., importin-beta) that recognizes a nuclear localization signal (NLS); importin-beta has surface FG-binding hydrophobic patches that allow it to dissolve transiently into the FG mesh and traverse the channel along with the cargo.",
            "Directionality comes from the asymmetric Ran nucleotide gradient: RanGTP is high in the nucleus (RanGEF/Rcc1 is nuclear), RanGDP is high in cytoplasm (RanGAP is cytoplasmic). RanGTP binds importin in the nucleus to release cargo; importin-RanGTP recycles back through the pore and then releases Ran upon GAP-stimulated hydrolysis in cytoplasm.",
        ],
        "rubric_notes": "A shallow answer says the pore lets in things with NLS sequences without explaining the FG-repeat hydrophobic mesh as the selectivity filter, the role of karyopherins as 'pass-through' carriers via FG-binding patches, or the Ran gradient that provides directionality.",
    },
    {
        "topic_bucket": "molecular_biology",
        "question": "How do intrinsically disordered proteins drive the formation of membraneless organelles (e.g., stress granules, P bodies) by liquid-liquid phase separation? Explain the mechanism step by step.",
        "reference_answer": [
            "Many RNA-binding and signaling proteins contain low-complexity intrinsically disordered regions (IDRs) enriched in repetitive aromatic, polar, or charged residues (e.g., FUS, TDP-43, hnRNP A1) that mediate weak multivalent interactions with each other and with RNA.",
            "Above a threshold concentration (and within a window of temperature, salt, and pH), the bulk solution becomes unstable to demixing; multivalent weak interactions (cation-pi, pi-pi stacking, electrostatic, hydrophobic) drive liquid-liquid phase separation into a dilute phase and a concentrated dense droplet phase.",
            "The droplet selectively concentrates particular components (clients) up to ~100-fold, accelerating biochemistry that requires multi-component encounters; the boundary is a sharp interface in composition, not a lipid bilayer, so it is permeable to small molecules but retains specific clients via affinity.",
            "Aberrant transitions from liquid to gel/aggregate (pathological mutations strengthening interactions, or aging/post-translational modifications) are implicated in neurodegenerative disease (e.g., FUS, TDP-43 in ALS); phase behavior is also regulated by RNA, chaperones, and protein modifications such as phosphorylation.",
        ],
        "rubric_notes": "A shallow answer says proteins clump into droplets without explaining LLPS mechanism: weak multivalent interactions between IDRs, threshold-concentration demixing, selective client enrichment, and pathological transitions from liquid to solid states.",
    },
    {
        "topic_bucket": "molecular_biology",
        "question": "How does a CRISPR-Cas9 ribonucleoprotein find and cleave a specific DNA target, and where do off-target cuts come from? Explain the mechanism step by step.",
        "reference_answer": [
            "Cas9 loaded with a guide RNA (sgRNA, with a 20-nt spacer) scans dsDNA and rejects non-PAM sites quickly; a successful PAM contact (NGG for SpCas9) triggers local DNA melting adjacent to the PAM and initiates spacer-protospacer base pairing.",
            "Spacer-protospacer pairing propagates from the PAM-proximal seed (about 8-12 nt closest to PAM) outward; mismatches in the seed region rapidly destabilize the R-loop and the complex dissociates without cutting.",
            "If pairing extends to a complete ~17-20 nt R-loop, conformational rearrangement activates the HNH and RuvC nuclease domains, which cut the target and non-target DNA strands respectively about 3 bp upstream of the PAM, producing a blunt-ended double-strand break.",
            "Off-target cuts arise when PAM-distal mismatches are tolerated and the R-loop completes despite imperfect pairing; engineered high-fidelity Cas9 variants destabilize the partial R-loop or raise the activation threshold, sacrificing some on-target rate to suppress off-targets.",
        ],
        "rubric_notes": "A shallow answer says Cas9 cuts at the guide-matched site without explaining the PAM-first scanning, the seed-region propagation of base pairing, the R-loop completeness threshold for nuclease activation, or how off-targets arise from PAM-distal mismatch tolerance.",
    },
    {
        "topic_bucket": "molecular_biology",
        "question": "How does the ribosome distinguish cognate from near-cognate aminoacyl-tRNAs to maintain low translation error rates? Explain the mechanism step by step.",
        "reference_answer": [
            "An EF-Tu-GTP-aminoacyl-tRNA ternary complex enters the ribosomal A site; the codon-anticodon contact triggers conformational changes in the small ribosomal subunit (closure of the decoding center, monitoring bases A1492/A1493/G530 inserting into the minor groove of the codon-anticodon helix) only when correct Watson-Crick geometry is achieved.",
            "The cognate-induced conformational change accelerates EF-Tu's GTP hydrolysis (the initial selection step); near-cognate tRNAs fail to induce the change efficiently and dissociate before hydrolysis, providing a first kinetic discrimination step.",
            "After GTP hydrolysis and EF-Tu-GDP release, the aminoacyl-tRNA must 'accommodate' from the A/T state into the peptidyl transferase center; cognate tRNAs accommodate quickly while near-cognate tRNAs accommodate more slowly and have additional opportunity to dissociate (proofreading step).",
            "The two-step kinetic discrimination compounds: each step can reject near-cognates with some probability, so combined error rates of 10^-3 to 10^-4 per codon are achieved despite individual base-pairing thermodynamics that would predict perhaps 10^-2.",
        ],
        "rubric_notes": "A shallow answer says the ribosome reads codons by base pairing without explaining the two-step kinetic proofreading: initial selection (gated by GTP hydrolysis after conformational verification of pairing geometry) followed by accommodation proofreading.",
    },
    {
        "topic_bucket": "molecular_biology",
        "question": "How does telomerase extend chromosome ends, and why is it usually repressed in somatic cells? Explain the mechanism step by step.",
        "reference_answer": [
            "Linear chromosomes have a problem at their ends: lagging-strand DNA replication leaves a 3' overhang that cannot be filled by conventional polymerases, so chromosomes shorten with each division (the end-replication problem).",
            "Telomerase is a ribonucleoprotein with a reverse transcriptase catalytic subunit (TERT) and an integral RNA component (TERC) that contains a short template sequence complementary to the telomeric repeat (TTAGGG in vertebrates).",
            "TERT binds the chromosome 3' overhang, anneals the template RNA region to it, and reverse-transcribes new repeat units onto the 3' end; after one repeat is synthesized, the enzyme translocates and re-anneals to add another (processive elongation), then conventional DNA polymerase fills in the complementary strand using a primase-laid primer.",
            "Most differentiated somatic cells repress telomerase expression; gradual telomere shortening with each division eventually triggers a DNA-damage response (replicative senescence or apoptosis), serving as a tumor-suppression mechanism. About 90% of cancers reactivate telomerase to bypass this limit, often via TERT promoter mutations.",
        ],
        "rubric_notes": "A shallow answer says telomerase rebuilds telomeres without explaining how a reverse transcriptase using its own RNA template extends only one strand (the 3' overhang) by repeated translocation, or the role of telomerase repression as a somatic-cell anti-cancer mechanism.",
    },

    # ---------------- geophysics (7) ----------------
    {
        "topic_bucket": "geophysics",
        "question": "What drives the Brewer-Dobson circulation in the stratosphere, and why does it transport air from the tropics to the poles? Explain the mechanism step by step.",
        "reference_answer": [
            "Tropospheric Rossby waves (planetary-scale waves driven by topography and land-sea heating contrasts) and gravity waves propagate upward into the stratosphere where the background flow is westerly in winter; their amplitude grows with altitude as density drops.",
            "When these waves reach altitudes where wave amplitudes saturate or wave-mean-flow critical levels are encountered, they break and dissipate, depositing easterly (westward) momentum into the winter mid-latitude stratospheric flow; this is the wave drag.",
            "By the conservation of angular momentum (the downward control principle), wave drag forces a poleward residual circulation in the stratosphere; mass continuity then requires upward motion in the tropics and downward motion at high latitudes.",
            "The circulation transports tropical-tropopause-layer air (with its low-ozone, anthropogenic-tracer signature) poleward and downward, distributing trace species globally; the strength of the circulation is seasonal, modulated by the QBO, ENSO, and predicted to evolve under climate change with implications for stratospheric water vapor and ozone recovery.",
        ],
        "rubric_notes": "A shallow answer says warm air rises in the tropics and sinks at the poles without identifying that the circulation is wave-driven (planetary wave breaking deposits easterly momentum) and that the resulting residual circulation is what produces tropical upwelling and polar downwelling.",
    },
    {
        "topic_bucket": "geophysics",
        "question": "Why does a layer of atomic sodium exist in the mesosphere at about 85-95 km altitude, and what limits its vertical extent? Explain the mechanism step by step.",
        "reference_answer": [
            "Meteoroids ablating in the upper mesosphere continuously deposit metallic vapor (Na, Fe, Mg, Ca) at altitudes around 80-110 km; this is the source of mesospheric metal layers.",
            "Below about 80 km, atomic Na reacts efficiently with O3 and H2O (Na + O3 -> NaO + O2; NaO + H2O -> NaOH; reactions also forming NaHCO3 and other compounds) and is removed from the atomic state; this sets the lower boundary of the layer.",
            "Above about 100 km, increasing solar EUV flux ionizes Na (Na + h-nu -> Na+ + e-) faster than recombination can reform atomic Na; ionic Na+ joins ionospheric chemistry rather than persisting as neutral atoms, setting the upper boundary.",
            "The narrow range where photochemistry maintains metal in the atomic state without ionizing it produces the sharply peaked layer; this layer is exploited for laser guide stars (Na D-line resonance fluorescence at 589 nm produces an artificial point source at about 90 km altitude for adaptive optics on large telescopes).",
        ],
        "rubric_notes": "A shallow answer says meteors deposit sodium high up without explaining what defines the layer's vertical extent: chemical loss to NaOH/NaHCO3 below and ionization losses above.",
    },
    {
        "topic_bucket": "geophysics",
        "question": "What drives the Atlantic Meridional Overturning Circulation (AMOC), and why is it sensitive to surface freshening at high northern latitudes? Explain the mechanism step by step.",
        "reference_answer": [
            "The Gulf Stream and North Atlantic Current carry warm salty water northward at the surface; the high salinity comes from net evaporation in subtropical and tropical Atlantic latitudes.",
            "At high northern latitudes (Labrador and Greenland-Iceland-Norwegian Seas), the surface water cools rapidly while retaining its high salinity; cooling raises density enough that surface water becomes denser than the underlying water and sinks (deep water formation), forming North Atlantic Deep Water.",
            "NADW flows southward at depth toward the Southern Ocean; wind-driven Ekman upwelling and diapycnal mixing eventually bring deep water back to the surface and the loop closes via a surface return flow, taking a millennium-scale time.",
            "Sinking is sensitive to surface salinity because near 0 C, density depends more on salinity than on temperature; freshwater input (Greenland melt, sea-ice melt, Arctic river runoff) reduces surface salinity and can suppress or stop sinking. A reduced AMOC weakens northward heat transport and cools the North Atlantic region; abrupt collapses are inferred from the paleoclimate record (Heinrich events).",
        ],
        "rubric_notes": "A shallow answer says cold dense water sinks in the North Atlantic without explaining the role of high salinity (carried from low-latitude evaporation) in providing the density needed to sink, or why freshening at the surface is what destabilizes the circulation.",
    },
    {
        "topic_bucket": "geophysics",
        "question": "How does a thundercloud build the voltage that drives a lightning flash? Explain the mechanism step by step.",
        "reference_answer": [
            "Inside the mixed-phase region of a cumulonimbus (about -10 to -25 C, where supercooled liquid water and ice coexist), strong updrafts loft small ice crystals while denser graupel (riming hail-like particles) tends to fall.",
            "When ice crystals collide with graupel in the presence of supercooled liquid water, charge is transferred between them: under typical mixed-phase conditions, the smaller ice crystals tend to take positive charge and the graupel takes negative charge (the non-inductive charging mechanism); the precise sign depends on temperature and liquid-water content (a sign reversal occurs near a critical 'charge reversal temperature').",
            "Updrafts carry positively charged ice crystals to the upper cloud; gravity carries negatively charged graupel to the middle and lower cloud; the charge separation produces a vertical dipole (and often a small lower positive region) reaching electric fields above 100 kV/m over kilometers.",
            "When the field exceeds local dielectric breakdown of moist air, a stepped leader propagates downward (or upward to ground), establishing an ionized channel; a return stroke neutralizes much of the charge, after which graupel-ice collisions rebuild the gradient until the next flash.",
        ],
        "rubric_notes": "A shallow answer says collisions in clouds separate charge without specifying the non-inductive ice-graupel charging in the mixed-phase region, the role of updrafts in spatially separating charges of different signs, or the breakdown threshold that triggers a flash.",
    },
    {
        "topic_bucket": "geophysics",
        "question": "How does the El Nino Southern Oscillation (ENSO) sustain its quasi-periodic switching between El Nino and La Nina states? Explain the mechanism step by step.",
        "reference_answer": [
            "In the neutral equatorial Pacific, easterly trade winds push warm surface water westward, creating a deep warm pool over Indonesia and tilted equatorial thermocline (deep in the west, shallow in the east); cold subsurface water upwells in the eastern Pacific, supporting the cold tongue.",
            "El Nino begins when a perturbation (e.g., a westerly wind burst) weakens the trade winds; warm pool water sloshes eastward as a downwelling Kelvin wave, deepening the eastern thermocline and suppressing upwelling. The warmer eastern Pacific further weakens the equatorial pressure gradient, weakening trades more (Bjerknes ocean-atmosphere positive feedback).",
            "The reverse process produces La Nina; the system also has memory in slow off-equatorial Rossby waves: during El Nino, off-equatorial Rossby waves carry shallow-thermocline anomalies westward, reflect at the western boundary as upwelling Kelvin waves, and propagate eastward to recharge cold subsurface water in the east.",
            "The recharge-discharge oscillator combines the fast Bjerknes feedback (months) with slow Rossby-Kelvin wave timescales (1-2 years), producing the 2-7 year ENSO cycle. ENSO interacts with mean-state ocean heat content and is modulated by background climate state; predictability is set by the fast feedback's stochastic forcing budget.",
        ],
        "rubric_notes": "A shallow answer says trade winds change and warm water sloshes around without identifying the Bjerknes feedback as the amplifier or the slow Rossby-wave-mediated recharge-discharge as the mechanism that sets the oscillation timescale.",
    },
    {
        "topic_bucket": "geophysics",
        "question": "What causes a sudden stratospheric warming, and why does it influence surface weather weeks later? Explain the mechanism step by step.",
        "reference_answer": [
            "During Northern winter, a strong westerly polar vortex circulates around the cold pole stratosphere; planetary-scale Rossby waves forced from the troposphere (orography, land-sea contrast, blocking patterns) propagate upward into the stratosphere where they can break.",
            "When tropospheric forcing produces unusually strong upward wave activity, breaking waves deposit large amounts of easterly momentum into the polar stratospheric flow, decelerating the vortex; if the deceleration is sufficient, the westerlies reverse to easterlies (a 'major' SSW), warm air from lower latitudes invades the polar stratosphere, and temperatures rise tens of K in days.",
            "The disrupted vortex weakens the stratospheric polar jet; the anomaly can propagate downward over weeks via wave-mean-flow interaction (the 'Northern Annular Mode' downward propagation), reaching the troposphere as a negative-NAM/negative-NAO state.",
            "The resulting tropospheric pattern often produces a weakened jet stream, equatorward shift of storm tracks, and cold-air outbreaks over Eurasia and eastern North America; these surface impacts can persist 1-2 months after the initial SSW (e.g., the 'Beast from the East' February 2018).",
        ],
        "rubric_notes": "A shallow answer says the stratosphere warms suddenly when the polar vortex breaks down without explaining that the forcing is upward-propagating tropospheric Rossby waves that break and decelerate the vortex, or how the anomaly's downward influence later changes surface weather.",
    },
    {
        "topic_bucket": "geophysics",
        "question": "Why does the F-region of the ionosphere persist through the night despite the absence of ionizing solar radiation? Explain the mechanism step by step.",
        "reference_answer": [
            "During day, solar EUV (about 30-100 nm wavelength) ionizes atomic oxygen and N2 in the upper atmosphere; the F-region peak (about 250-400 km) forms where the production rate balances loss, with the highest electron density in the daytime.",
            "Recombination of O+ at F-region altitudes proceeds via the slow two-step pathway O+ + N2 -> NO+ + N (or O+ + O2 -> O2+ + O), followed by fast dissociative recombination NO+ + e -> N + O; the rate-limiting first step has a low rate constant, so the recombination time is hours rather than seconds.",
            "Because the recombination time is comparable to or longer than the duration of darkness at most latitudes, ionization persists overnight; thermospheric neutral winds and electric fields also lift plasma to higher altitudes where recombination is even slower, prolonging the nighttime layer.",
            "Equatorial fountain dynamics (E x B drift lifting plasma over the magnetic equator and downward diffusion to mid-latitudes along field lines) helps sustain low-latitude nighttime ionization; geomagnetic storms inject energy that disrupts the F-layer, causing communications blackouts and GPS errors.",
        ],
        "rubric_notes": "A shallow answer says the ionosphere remains ionized at night because recombination is slow without identifying that the slowness comes from the bottleneck step of charge transfer to molecular ions before dissociative recombination, or describing the role of thermospheric winds and equatorial fountain in sustaining the layer.",
    },

    # ---------------- engineering (6) ----------------
    {
        "topic_bucket": "engineering",
        "question": "Why is sealing the apex tips of a Wankel (rotary) engine's rotor against its housing fundamentally harder than sealing piston rings, and what fails? Explain the mechanism step by step.",
        "reference_answer": [
            "A Wankel rotor is a triangular geometry that rotates eccentrically inside an epitrochoid-shaped housing; each of the three apex tips sweeps a continuously curving rubbing path along the housing across all engine cycles (intake, compression, ignition, expansion, exhaust).",
            "An apex seal is a thin spring-loaded strip mounted in a slot at each rotor tip; centrifugal force and combustion gas pressure press the seal outward against the housing surface, and the seal tip rubs continuously across the housing as the rotor turns. The seal must conform to local housing curvature changes during the cycle.",
            "Sealing also requires good corner geometry: at each rotor face boundary, the apex seal meets two side seals, and the meeting points (corner seal pieces) must remain gas-tight in three dimensions even under thermal expansion mismatch between rotor and housing, which is large because the rotor sees combustion gas while the housing is water-cooled.",
            "Failure modes include rapid apex-seal tip wear (loss of sealing force and compression), seal-corner blowby (gas leaking past corner pieces), and housing scoring (chatter marks from seal vibration); modern designs use ceramic-coated or composite seals and chrome-plated/coated housings to extend service life, but apex-seal life remains the dominant durability constraint of rotary engines.",
        ],
        "rubric_notes": "A shallow answer says rotary engines have apex seals that wear without explaining why sealing is hard (continuous rubbing along a curving epitrochoid path under combustion conditions) or what specifically fails (tip wear, corner-seal blowby, housing scoring).",
    },
    {
        "topic_bucket": "engineering",
        "question": "How does an active magnetic bearing levitate a rotating shaft and remain stable despite the inherent instability of attractive electromagnetic forces? Explain the mechanism step by step.",
        "reference_answer": [
            "An active magnetic bearing places electromagnets around the shaft (or rotor sleeve); permanent magnets and/or DC bias current in the coils provide a baseline attractive force that supports much of the load. Earnshaw's theorem implies static electromagnetic levitation is unstable, so feedback control is required.",
            "Position sensors (eddy-current or capacitive, kHz-MHz bandwidth) measure shaft displacement from the centered position; a controller (typically PID with notch filters at known rotor mode frequencies) computes corrective currents in each electromagnet to push the shaft back toward center.",
            "Currents flow through the coils faster than the rotor can drift away from center (control bandwidth must exceed the unstable mode frequency by a comfortable margin); the closed-loop system is stable as long as the controller's gains and phase margin are correctly designed for the rotor's mass, gyroscopic coupling, and modal dynamics.",
            "Failure modes include power loss (the rotor falls onto catcher bearings designed to absorb the touchdown), sensor failure (control loop gets bad position data and can amplify rather than damp motion), and rotor unbalance forces exceeding the actuator's force limit; advantages of magnetic bearings are no contact, very high speeds, no lubrication, and tunable damping.",
        ],
        "rubric_notes": "A shallow answer says magnetic bearings use sensors and electromagnets to levitate the shaft without identifying that static attractive levitation is unstable (Earnshaw) and that the bearing depends fundamentally on closed-loop control bandwidth exceeding the unstable mode frequencies.",
    },
    {
        "topic_bucket": "engineering",
        "question": "How does a hydraulic ram pump lift water uphill using only the energy of flowing source water, and why is it cyclic? Explain the mechanism step by step.",
        "reference_answer": [
            "Source water at a moderate elevation flows down a long drive pipe to the ram; the column of water accelerates under gravity until it reaches a high velocity at the ram inlet, where an open waste valve releases the water out of the bottom of the ram.",
            "As flow speed rises, hydrodynamic drag on the waste valve disc grows quadratically and at a critical speed the disc snaps shut against its seat; the moving water column abruptly decelerates, generating a hydraulic-hammer pressure spike (high pressure short pulse).",
            "The pressure spike forces a one-way check valve open into a sealed air chamber (Windkessel); some water enters the air chamber, compressing the trapped air, and continues out through the delivery pipe to the higher destination. As the spike subsides, the check valve closes and the air chamber discharges its stored energy smoothly into the delivery flow during the lull.",
            "After the spike, pressure drops below the static head and the waste valve falls open under gravity (or spring); water resumes flowing out through the waste valve, the cycle repeats at about 0.5-2 Hz. Volumetric efficiency is typically 10-25% (most water exits through the waste valve), but no external power is required and the device runs unattended for years.",
        ],
        "rubric_notes": "A shallow answer says the pump uses water flow to push water up without identifying the hydraulic-hammer pressure spike (waste-valve closure stops the water column) and the air chamber as the mechanism that uses that spike to drive water uphill.",
    },
    {
        "topic_bucket": "engineering",
        "question": "How does a heat pipe transport heat with an effective conductivity orders of magnitude higher than a copper rod of the same dimensions? Explain the mechanism step by step.",
        "reference_answer": [
            "A heat pipe is a sealed tube containing a small amount of working fluid (water for moderate temperatures, ammonia for low, sodium for high) and a wick (sintered powder, mesh, or grooves) lining the inner wall; the fluid is at saturation pressure for the operating temperature, so liquid and vapor coexist throughout the pipe.",
            "At the hot (evaporator) end, heat flux into the pipe boils liquid in the wick, forming vapor; the vaporization absorbs latent heat from the wall. The pressure rises slightly at the evaporator end, creating a small pressure gradient driving vapor down the central core toward the cold end.",
            "At the cold (condenser) end, vapor condenses on the wick, releasing latent heat to the wall; the condensate is pulled back along the wick toward the evaporator by capillary action (the meniscus curvature in the wick provides the driving pressure to overcome friction and gravity).",
            "Heat is transported as latent enthalpy of phase change rather than conducted through solid metal; effective conductivity along the pipe is hundreds to thousands of times that of copper. Failure modes include capillary dryout (heat input exceeds the wick's ability to return liquid), incondensable gas accumulation at the cold end (blocks the condenser), and freezing of the working fluid below operating range.",
        ],
        "rubric_notes": "A shallow answer says the pipe's working fluid evaporates and condenses to carry heat without explaining that the high effective conductivity comes from latent-heat transport in the vapor and that condensate return is by capillary pumping in the wick (no external pump needed).",
    },
    {
        "topic_bucket": "engineering",
        "question": "How does a cesium-beam atomic clock lock its local oscillator to the Cs hyperfine transition frequency, and what makes this lock long-term stable? Explain the mechanism step by step.",
        "reference_answer": [
            "A microwave local oscillator (LO) at about 9.192 631 770 GHz drives a Ramsey cavity; cesium atoms from a thermal beam (or laser-cooled fountain) pass through the cavity. The Ramsey configuration uses two separated microwave interaction regions with a flight gap between, producing narrow interference fringes in the atomic transition probability versus LO frequency.",
            "After the cavity, a state-selective detector counts atoms in one hyperfine state; the count rate is highest when the LO frequency matches the central Ramsey fringe, and decreases sharply on either side. The fringe width is set by the inverse flight time between the two microwave interaction regions.",
            "A servo loop dithers the LO frequency slightly above and below its setpoint, observes the corresponding modulation in detected atom flux, and computes a correction signal proportional to the slope of the fringe; this correction is integrated to update the LO frequency, locking it to the fringe peak.",
            "Long-term stability inherits the invariance of the Cs hyperfine transition; the LO drifts of free-running quartz are continuously corrected toward the atomic frequency. The relative frequency stability of a primary Cs standard is typically 10^-13 to 10^-15, set by atomic shot noise, fringe contrast, and systematic shifts (magnetic field, blackbody, gravitational redshift) that must be modeled and removed.",
        ],
        "rubric_notes": "A shallow answer says the local oscillator is locked to the cesium transition without explaining the Ramsey-fringe interrogation, the dither-and-detect feedback that converts atom counts to a frequency correction, or why this scheme transfers atomic invariance to the LO.",
    },
    {
        "topic_bucket": "engineering",
        "question": "How does a closed-loop adaptive optics system correct atmospheric wavefront distortion in real time on a large telescope? Explain the mechanism step by step.",
        "reference_answer": [
            "Atmospheric turbulence introduces phase distortions across the telescope aperture; uncorrected, these scramble the diffraction-limited PSF into a speckle pattern with a characteristic atmospheric coherence length r0 (a few cm at visible, tens of cm at infrared). Adaptive optics aims to flatten the wavefront before science detection.",
            "A wavefront sensor (Shack-Hartmann lenslet array or pyramid sensor) samples the local tilt of the wavefront across many subapertures from a guide star; tilt measurements are converted to a discrete map of wavefront phase distortion across the aperture.",
            "A reconstructor (matrix inversion, often Tikhonov-regularized; modal decomposition such as Karhunen-Loeve or Zernike) computes the deformable-mirror actuator commands needed to flatten the residual wavefront after correction; commands are sent to a deformable mirror with hundreds to thousands of voice-coil or piezoelectric actuators behind a thin reflective faceplate.",
            "The loop runs at kHz rates to track the atmosphere's Greenwood frequency; loop bandwidth, sensor photon noise, and total latency (sense-compute-actuate) limit the achievable Strehl ratio. For science targets too faint for natural guide stars, sodium-layer laser guide stars provide an artificial reference for high-order correction (with a separate natural guide star for tip-tilt, which lasers cannot sense due to round-trip propagation).",
        ],
        "rubric_notes": "A shallow answer says a deformable mirror corrects atmospheric distortion without explaining the wavefront sensor (Shack-Hartmann tilt measurements at subapertures), the reconstructor that converts measurements to actuator commands, the kHz loop rate set by the Greenwood frequency, or the laser guide star tip-tilt limitation.",
    },
]

# split assignment per topic_bucket: 26 train / 14 eval, deterministic by position within bucket
SPLITS_PER_BUCKET = {
    "instrumentation": (4, 3),     # 7 items
    "manufacturing": (5, 2),       # 7 items
    "electrochemistry": (4, 2),    # 6 items
    "molecular_biology": (5, 2),   # 7 items
    "geophysics": (4, 3),          # 7 items
    "engineering": (4, 2),         # 6 items
}


def main() -> None:
    out_items: list[dict] = []
    bucket_index: dict[str, int] = {}
    for idx, item in enumerate(ITEMS, start=1):
        bucket = item["topic_bucket"]
        position = bucket_index.get(bucket, 0)
        bucket_index[bucket] = position + 1
        n_train, n_eval = SPLITS_PER_BUCKET[bucket]
        if position >= n_train + n_eval:
            raise RuntimeError(f"bucket {bucket} has more items than allotted")
        split = "train" if position < n_train else "eval"
        out_items.append(
            {
                "item_id": f"mechH_{idx:03d}",
                "topic_bucket": bucket,
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "rubric_notes": item["rubric_notes"],
                "split": split,
            }
        )

    counts = {"train": 0, "eval": 0}
    for it in out_items:
        counts[it["split"]] += 1
    assert counts == {"train": 26, "eval": 14}, counts

    with OUT_PATH.open("w") as f:
        for it in out_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    sha = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
    math_sha = hashlib.sha256(MATH_PATH.read_bytes()).hexdigest()
    mech_sha = hashlib.sha256(MECH_PATH.read_bytes()).hexdigest()
    HASHES_PATH.write_text(
        f"{math_sha}  math.jsonl\n"
        f"{mech_sha}  mechanism_qa.jsonl\n"
        f"{sha}  mechanism_qa_harder.jsonl\n"
    )

    by_bucket = {}
    for it in out_items:
        by_bucket.setdefault(it["topic_bucket"], {"train": 0, "eval": 0})[it["split"]] += 1
    print(f"wrote {len(out_items)} items to {OUT_PATH.relative_to(REPO)}")
    for b in sorted(by_bucket):
        print(f"  {b:<20} train={by_bucket[b]['train']} eval={by_bucket[b]['eval']}")
    print(f"\nSHA256 mechanism_qa_harder.jsonl  {sha}")


if __name__ == "__main__":
    main()
